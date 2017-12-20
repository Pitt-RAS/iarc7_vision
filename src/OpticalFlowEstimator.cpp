#include "iarc7_vision/OpticalFlowEstimator.hpp"

// BAD HEADER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Geometry>
#pragma GCC diagnostic pop
// END BAD HEADER

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "iarc7_vision/cv_utils.hpp"
#include <ros_utils/SafeTransformWrapper.hpp>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <iarc7_msgs/OrientationAnglesStamped.h>

namespace iarc7_vision {

OpticalFlowEstimator::OpticalFlowEstimator(
        const OpticalFlowEstimatorSettings& flow_estimator_settings,
        const OpticalFlowDebugSettings& debug_settings)
    : flow_estimator_settings_(flow_estimator_settings),
      debug_settings_(debug_settings),
      have_valid_last_image_(false),
      images_skipped_(0),
      last_scaled_image_(),
      last_scaled_grayscale_image_(),
      transform_wrapper_(),
      current_altitude_(0.0),
      current_orientation_(),
      last_orientation_(),
      current_camera_to_level_quad_tf_(),
      last_camera_to_level_quad_tf_(),
      last_message_time_(),
      expected_input_size_(),
      target_size_(),
      local_nh_("optical_flow_estimator"),
      debug_average_velocity_vector_image_pub_(
              local_nh_.advertise<sensor_msgs::Image>("average_vector_image",
                                                     1)),
      debug_camera_rel_raw_pub_(
              local_nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
                  "twist_camera_relative", 10)),
      debug_correction_pub_(
              local_nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
                  "twist_correction", 10)),
      debug_raw_pub_(
              local_nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
                  "twist_raw", 10)),
      debug_unrotated_vel_pub_(
              local_nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
                  "twist_unrotated", 10)),
      debug_velocity_vector_image_pub_(
              local_nh_.advertise<sensor_msgs::Image>("vector_image", 1)),
      orientation_pub_(
              local_nh_.advertise<iarc7_msgs::OrientationAnglesStamped>(
                  "orientation", 10)),
      twist_pub_(
              local_nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
                  "twist", 10))
{
}

bool __attribute__((warn_unused_result))
        OpticalFlowEstimator::onSettingsChanged()
{
    cv::Size new_target_size;
    new_target_size.width = expected_input_size_.width
                          * flow_estimator_settings_.scale_factor;
    new_target_size.height = expected_input_size_.height
                           * flow_estimator_settings_.scale_factor;

    if (expected_input_size_ != cv::Size(0, 0)
     && (new_target_size.width == 0 || new_target_size.height == 0)) {
        ROS_ERROR_STREAM("Target size is zero after updating ("
                      << new_target_size
                      << ")");
        return false;
    }

    target_size_ = new_target_size;

    if (expected_input_size_ != cv::Size(0, 0)) {
        // Note: neither of the outputs can be the input image, so we need to
        // do this copying thing
        cv::gpu::GpuMat scaled_image;
        resizeAndConvertImages(last_scaled_image_,
                               scaled_image,
                               last_scaled_grayscale_image_);
        last_scaled_image_ = scaled_image;
    }

    return true;
}

void OpticalFlowEstimator::update(const sensor_msgs::Image::ConstPtr& message)
{
    if (cv::gpu::getCudaEnabledDeviceCount() == 0) {
        ROS_ERROR_ONCE("Unable to run OpticalFlow without CUDA");
        have_valid_last_image_ = false;
        return;
    }

    // start time for debugging time spent in updateFilteredPosition
    const ros::WallTime start = ros::WallTime::now();

    // make sure our current position is up to date
    if (!updateFilteredPosition(
                message->header.stamp,
                ros::Duration(flow_estimator_settings_.tf_timeout))) {
        ROS_ERROR("Unable to update position for optical flow");
        return;
    }

    if (debug_settings_.debug_times) {
        ROS_WARN_STREAM("updateFilteredPosition: "
                     << ros::WallTime::now() - start);
    }

    // Make sure we're in an allowed position to calculate optical flow
    if (!canEstimateFlow()) {
        return;
    }

    const boost::shared_ptr<const cv_bridge::CvImage> curr_image_msg
        = cv_bridge::toCvShare(message);
    const cv::Mat& curr_image = curr_image_msg->image;

    if (have_valid_last_image_) {
        if (curr_image.size() != expected_input_size_) {
            ROS_ERROR("Ignoring image of size (%dx%d), expected (%dx%d)",
                      curr_image.size().width,
                      curr_image.size().height,
                      expected_input_size_.width,
                      expected_input_size_.height);
            return;
        }

        try {
            // Scale and convert input image
            cv::gpu::GpuMat curr_gpu_image(curr_image);
            cv::gpu::GpuMat scaled_image;
            cv::gpu::GpuMat scaled_gray_image;

            resizeAndConvertImages(curr_gpu_image,
                                   scaled_image,
                                   scaled_gray_image);

            // Get velocity estimate from average vector
            processImage(scaled_image,
                         scaled_gray_image,
                         message->header.stamp,
                         images_skipped_ == 0);
            images_skipped_ = (images_skipped_ + 1)
                           % (flow_estimator_settings_.debug_frameskip + 1);

            last_scaled_image_ = scaled_image;
            last_scaled_grayscale_image_ = scaled_gray_image;
        } catch (const std::exception& ex) {
            ROS_ERROR_STREAM("Caught exception processing image flow: "
                          << ex.what());
            have_valid_last_image_ = false;
        }
    } else {
        expected_input_size_ = curr_image.size();
        last_scaled_image_.upload(curr_image);
        ROS_ASSERT(onSettingsChanged());
        have_valid_last_image_ = true;
    }

    last_message_time_ = message->header.stamp;
    last_orientation_ = current_orientation_;
    last_camera_to_level_quad_tf_ = current_camera_to_level_quad_tf_;
}

bool OpticalFlowEstimator::waitUntilReady(
        const ros::Duration& startup_timeout)
{
    return updateFilteredPosition(ros::Time::now(), startup_timeout);
}

bool OpticalFlowEstimator::canEstimateFlow() const
{
    if (current_altitude_ < flow_estimator_settings_.min_estimation_altitude) {
        ROS_WARN_THROTTLE(2.0,
                          "Height (%f) is below min processing height (%f)",
                          current_altitude_,
                          flow_estimator_settings_.min_estimation_altitude);
        return false;
    }

    geometry_msgs::Vector3Stamped camera_forward_vector;
    camera_forward_vector.vector.x = 0;
    camera_forward_vector.vector.y = 0;
    camera_forward_vector.vector.z = 1;
    tf2::doTransform(camera_forward_vector,
                     camera_forward_vector,
                     current_camera_to_level_quad_tf_);
    if (camera_forward_vector.vector.z
            > -flow_estimator_settings_.camera_vertical_threshold) {
        ROS_WARN_THROTTLE(
                2.0,
                "Camera is not close enough to vertical to calculate flow");
        return false;
    }

    return true;
}

geometry_msgs::TwistWithCovarianceStamped
        OpticalFlowEstimator::estimateVelocityFromFlowVector(
                const cv::Point2f& average_vec,
                const ros::Time& time) const
{
    // Get the pitch and roll of the camera in euler angles
    // NOTE: CAMERA FRAME CONVENTIONS ARE DIFFERENT, SEE REP103
    // http://www.ros.org/reps/rep-0103.html
    double yaw, pitch, roll;
    getYPR(current_orientation_, yaw, pitch, roll);

    double last_yaw, last_pitch, last_roll;
    getYPR(last_orientation_, last_yaw, last_pitch, last_roll);

    // Calculate time between last and current frame
    double dt = (time - last_message_time_).toSec();

    // Publish the orientation we're using for debugging purposes
    if (debug_settings_.debug_orientation) {
        iarc7_msgs::OrientationAnglesStamped ori_msg;
        ori_msg.header.stamp = time;
        ori_msg.data.pitch = pitch;
        ori_msg.data.roll = roll;
        ori_msg.data.yaw = yaw;
        orientation_pub_.publish(ori_msg);
    }

    // Distance from the camera to the ground plane, along the camera's +z axis
    //
    // Calculation based on a right triangle with one vertex at the camera,
    // one vertex on the ground directly below the camera, and one vertex at
    // the intersection of the camera's forward vector with the ground.  This
    // calculates the hypotenuse if the vertical leg has length
    // current_altitude_ and the horizontal leg has length
    // ((current_altitude_*tan(pitch))^2 + (current_altitude_*tan(roll))^2)^0.5
    double distance_to_plane = current_altitude_
                             * std::sqrt(1.0
                                       + std::pow(std::tan(pitch), 2.0)
                                       + std::pow(std::tan(roll), 2.0));

    // Multiplier that converts measurements in pixels to measurements in
    // meters in the plane parallel to the camera's xy plane and going through
    // the point Pg, where Pg is intersection between the camera's +z axis and
    // the ground plane
    //
    // Focal length gives distance from camera to image plane in pixels, so
    // dividing distance to plane by this number gives the multiplier we want
    double current_meters_per_px = distance_to_plane
                         / getFocalLength(target_size_,
                                          flow_estimator_settings_.fov);

    // Calculate the average velocity in the level camera frame (i.e. camera
    // frame if our pitch and roll were zero)
    // Note that cos(roll) is negative, because the z axis is down
    cv::Point2f velocity_uncorrected;
    velocity_uncorrected.x = (-average_vec.x * current_meters_per_px)
                           / std::cos(pitch)
                           / dt;
    velocity_uncorrected.y = (-average_vec.y * current_meters_per_px)
                           / -std::cos(roll)
                           / dt;

    double dp;
    double dr;

    // These two if statements make sure that dp and dr are the shortest change
    // in angle that would produce the new observed orientation
    //
    // i.e. a change from 0.1rad to (2pi-0.1)rad should result in a delta of
    // -0.2rad, not (2pi-0.2)rad
    if (last_pitch > CV_PI/2 && pitch < -CV_PI/2) {
        dp = (pitch + 2*CV_PI - last_pitch);
    } else if (last_pitch < -CV_PI/2 && pitch > CV_PI/2) {
        dp = (pitch - last_pitch - 2*CV_PI);
    } else {
        dp = (pitch - last_pitch);
    }

    if (last_roll > CV_PI/2 && roll < -CV_PI/2) {
        dr = (roll + 2*CV_PI - last_roll);
    } else if (last_roll < -CV_PI/2 && roll > CV_PI/2) {
        dr = (roll - last_roll - 2*CV_PI);
    } else {
        dr = (roll - last_roll);
    }

    double dpitch_dt = dp / dt;
    double droll_dt = dr / dt;

    // Observed velocity in camera frame due to rotation
    //
    // Positive dpdt means that the x vector rotates down, which means we think
    // we're moving in the -x direction
    //
    // Positive drdt means that the y vector rotates down, which means we think
    // we're moving in the -y direction
    //
    // Note again that cos(roll) is negative, because the z axis is down
    cv::Point2f correction_vel;
    correction_vel.x = distance_to_plane
                     * -dpitch_dt
                     / std::cos(pitch);
    correction_vel.y = distance_to_plane
                     * -droll_dt
                     / -std::cos(roll);

    // Actual velocity in level camera frame (i.e. camera frame if our pitch
    // and roll were zero)
    Eigen::Vector3d corrected_vel(
        velocity_uncorrected.x - correction_vel.x,
        velocity_uncorrected.y - correction_vel.y,
        0.0);

    last_pitch = pitch;
    last_roll = roll;

    // Calculate covariance
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    covariance(0, 0) = std::pow(flow_estimator_settings_.variance_scale
                              * dpitch_dt, 2.0)
                      + flow_estimator_settings_.variance;
    covariance(1, 1) = std::pow(flow_estimator_settings_.variance_scale
                              * droll_dt, 2.0)
                      + flow_estimator_settings_.variance;

    // Rotation matrix from level camera frame to level_quad
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())
                    * Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ());

    // Get velocity and covariance in level_quad frame
    Eigen::Vector3d level_quad_vel = rotation_matrix * corrected_vel;
    Eigen::Matrix3d level_quad_covariance = rotation_matrix
                                          * covariance
                                          * rotation_matrix.inverse();

    // Correct for movement of camera frame relative to level_quad
    //
    // When the drone rotates the camera has some velocity relative to the
    // ground even if the center of the drone doesn't move relative to the
    // ground, this cancels that effect
    geometry_msgs::PointStamped curr_pos, last_pos;
    tf2::doTransform(curr_pos, curr_pos, current_camera_to_level_quad_tf_);
    tf2::doTransform(last_pos, last_pos, last_camera_to_level_quad_tf_);
    double camera_relative_vel_x = (curr_pos.point.x - last_pos.point.x) / dt;
    double camera_relative_vel_y = (curr_pos.point.y - last_pos.point.y) / dt;
    level_quad_vel.x() -= camera_relative_vel_x;
    level_quad_vel.y() -= camera_relative_vel_y;

    // Fill out the twist
    geometry_msgs::TwistWithCovarianceStamped twist;
    twist.header.stamp = time;
    twist.header.frame_id = "level_quad";
    twist.twist.twist.linear.x = level_quad_vel.x();
    twist.twist.twist.linear.y = level_quad_vel.y();
    twist.twist.twist.linear.z = 0.0;

    twist.twist.covariance[0] = level_quad_covariance(0, 0);
    twist.twist.covariance[1] = level_quad_covariance(0, 1);
    twist.twist.covariance[6] = level_quad_covariance(1, 0);
    twist.twist.covariance[7] = level_quad_covariance(1, 1);

    // Publish intermediate twists for debugging
    if (debug_settings_.debug_intermediate_velocities) {
        geometry_msgs::TwistWithCovarianceStamped twist_correction = twist;
        twist_correction.twist.twist.linear.x = correction_vel.x;
        twist_correction.twist.twist.linear.y = correction_vel.y;
        debug_correction_pub_.publish(twist_correction);

        geometry_msgs::TwistWithCovarianceStamped twist_raw = twist;
        twist_raw.twist.twist.linear.x = velocity_uncorrected.x;
        twist_raw.twist.twist.linear.y = velocity_uncorrected.y;
        debug_raw_pub_.publish(twist_raw);

        geometry_msgs::TwistWithCovarianceStamped twist_unrotated = twist;
        twist_unrotated.twist.twist.linear.x = corrected_vel.x();
        twist_unrotated.twist.twist.linear.y = corrected_vel.y();
        debug_unrotated_vel_pub_.publish(twist_unrotated);

        geometry_msgs::TwistWithCovarianceStamped twist_camera_rel = twist;
        twist_camera_rel.twist.twist.linear.x = camera_relative_vel_x;
        twist_camera_rel.twist.twist.linear.y = camera_relative_vel_y;
        debug_camera_rel_raw_pub_.publish(twist_camera_rel);
    }

    return twist;
}

bool OpticalFlowEstimator::findAverageVector(
        const std::vector<cv::Point2f>& tails,
        const std::vector<cv::Point2f>& heads,
        const std::vector<uchar>& status,
        const double x_cutoff,
        const double y_cutoff,
        const cv::Size& image_size,
        cv::Point2f& average)
{
    double total_x = 0.0;
    double total_y = 0.0;
    size_t num_points = 0;

    // TODO filter properly based on magnitude, angle and standard deviation
    for (size_t i = 0; i < tails.size(); ++i) {
        if (status[i]
         && tails[i].x > image_size.width  * x_cutoff
         && tails[i].x < image_size.width  * (1.0 - x_cutoff)
         && tails[i].y > image_size.height * y_cutoff
         && tails[i].y < image_size.height * (1.0 - y_cutoff)) {

            cv::Point tail = tails[i];
            cv::Point head = heads[i];
            total_x += head.x - tail.x;
            total_y += head.y - tail.y;
            num_points++;
        }
    }

    if (num_points != 0) {
        average.x = total_x / static_cast<double>(num_points);
        average.y = total_y / static_cast<double>(num_points);
    }

    return num_points != 0;
}

void OpticalFlowEstimator::findFeatureVectors(
        const cv::gpu::GpuMat& curr_frame,
        const cv::gpu::GpuMat& /*curr_gray_frame*/,
        const cv::gpu::GpuMat& /*last_frame*/,
        const cv::gpu::GpuMat& last_gray_frame,
        std::vector<cv::Point2f>& tails,
        std::vector<cv::Point2f>& heads,
        std::vector<uchar>& status,
        bool debug) const
{
    const ros::WallTime start = ros::WallTime::now();

    // Create the feature detector and perform feature detection
    cv::gpu::GoodFeaturesToTrackDetector_GPU detector(
                    flow_estimator_settings_.points,
                    flow_estimator_settings_.quality_level,
                    flow_estimator_settings_.min_dist);
    cv::gpu::GpuMat d_prev_pts;

    detector(last_gray_frame, d_prev_pts);

    if (debug_settings_.debug_times) {
        ROS_WARN_STREAM("post detector: " << ros::WallTime::now() - start);
    }

    // Create optical flow object
    cv::gpu::PyrLKOpticalFlow d_pyrLK;

    d_pyrLK.winSize.width = flow_estimator_settings_.win_size;
    d_pyrLK.winSize.height = flow_estimator_settings_.win_size;
    d_pyrLK.maxLevel = flow_estimator_settings_.max_level;
    d_pyrLK.iters = flow_estimator_settings_.iters;

    cv::gpu::GpuMat d_next_pts;
    cv::gpu::GpuMat d_status;

    d_pyrLK.sparse(last_scaled_image_,
                   curr_frame,
                   d_prev_pts,
                   d_next_pts,
                   d_status);

    if (debug_settings_.debug_times) {
        ROS_WARN_STREAM("PYRLK sparse: " << ros::WallTime::now() - start);
    }

    cv_utils::downloadVector(d_prev_pts, tails);
    cv_utils::downloadVector(d_next_pts, heads);
    cv_utils::downloadVector(d_status, status);

    // Publish debugging image with all vectors drawn
    if (debug && debug_settings_.debug_vectors_image) {
        // Draw arrows
        cv::Mat arrow_image;
        curr_frame.download(arrow_image);
        cv_utils::drawArrows(arrow_image,
                             tails,
                             heads,
                             status,
                             cv::Scalar(255, 0, 0));

        cv_bridge::CvImage cv_image {
            std_msgs::Header(),
            sensor_msgs::image_encodings::RGBA8,
            arrow_image
        };

        debug_velocity_vector_image_pub_.publish(cv_image.toImageMsg());
    }
}

double OpticalFlowEstimator::getFocalLength(
        const cv::Size& img_size, double fov)
{
    // Calculates distance from image center to corner, divides by tan of half
    // the FOV to get distance from focal point to center of image plane
    return std::hypot(img_size.width/2.0, img_size.height/2.0)
         / std::tan(fov / 2.0);
}

void OpticalFlowEstimator::getYPR(const tf2::Quaternion& orientation,
                                  double& y,
                                  double& p,
                                  double& r)
{
    tf2::Matrix3x3 matrix;
    matrix.setRotation(orientation);
    matrix.getEulerYPR(y, p, r);
}

void OpticalFlowEstimator::processImage(const cv::gpu::GpuMat& image,
                                        const cv::gpu::GpuMat& gray_image,
                                        const ros::Time& time,
                                        bool debug) const
{
    // Find vectors from image
    std::vector<cv::Point2f> tails;
    std::vector<cv::Point2f> heads;
    std::vector<uchar> status;
    findFeatureVectors(image,
                       gray_image,
                       last_scaled_image_,
                       last_scaled_grayscale_image_,
                       tails,
                       heads,
                       status,
                       debug);

    // Calculate the average movement of the features in the camera frame
    cv::Point2f average_vec;
    bool found_average = findAverageVector(
            tails,
            heads,
            status,
            flow_estimator_settings_.x_cutoff_region_velocity_measurement,
            flow_estimator_settings_.y_cutoff_region_velocity_measurement,
            target_size_,
            average_vec);

    if (!found_average) {
        ROS_ERROR("OpticalFlow image with no valid features, returning");
        return;
    }

    // Publish debugging image with average vector drawn
    if (debug && debug_settings_.debug_average_vector_image) {
        cv::Mat arrow_image;
        last_scaled_image_.download(arrow_image);

        const cv::Point2f start_point(target_size_.width / 2.0,
                                      target_size_.height / 2.0);

        const cv::Point2f end_point(average_vec.x + start_point.x,
                                    average_vec.y + start_point.y);

        cv_utils::drawArrows(arrow_image,
                             { start_point },
                             { end_point },
                             { 1 },
                             cv::Scalar(255, 0, 0));

        const cv_bridge::CvImage cv_image {
            std_msgs::Header(),
            sensor_msgs::image_encodings::RGBA8,
            arrow_image
        };

        debug_average_velocity_vector_image_pub_.publish(
                cv_image.toImageMsg());
    }

    const geometry_msgs::TwistWithCovarianceStamped velocity
        = estimateVelocityFromFlowVector(average_vec,
                                         time);

    if (!std::isfinite(velocity.twist.twist.linear.x)
     || !std::isfinite(velocity.twist.twist.linear.y)
     || !std::isfinite(velocity.twist.covariance[0])    // variance of x
     || !std::isfinite(velocity.twist.covariance[1])    // covariance of x & y
     || !std::isfinite(velocity.twist.covariance[6])    // covariance of x & y
     || !std::isfinite(velocity.twist.covariance[7])) { // variance of y
        ROS_ERROR_STREAM("Invalid measurement in OpticalFlowEstimator: "
                      << velocity);
    } else {
        // Publish velocity estimate
        twist_pub_.publish(velocity);
    }
}

void OpticalFlowEstimator::resizeAndConvertImages(const cv::gpu::GpuMat& image,
                                                  cv::gpu::GpuMat& scaled,
                                                  cv::gpu::GpuMat& gray) const
{
    const ros::WallTime start = ros::WallTime::now();

    cv::gpu::resize(image,
                    scaled,
                    target_size_);

    if (debug_settings_.debug_times) {
        ROS_WARN_STREAM("post resize: " << ros::WallTime::now() - start);
    }

    cv::gpu::cvtColor(scaled,
                      gray,
                      CV_RGBA2GRAY);

    if (debug_settings_.debug_times) {
        ROS_WARN_STREAM("post cvtColor: " << ros::WallTime::now() - start);
    }
}

bool OpticalFlowEstimator::updateFilteredPosition(const ros::Time& time,
                                                  const ros::Duration& timeout)
{
    geometry_msgs::TransformStamped filtered_position_transform_stamped;
    geometry_msgs::TransformStamped camera_to_level_quad_tf_stamped;

    bool success = transform_wrapper_.getTransformAtTime(
            filtered_position_transform_stamped,
            "map",
            "bottom_camera_optical",
            time,
            timeout);

    if (!success) {
        return false;
    }

    success = transform_wrapper_.getTransformAtTime(
            camera_to_level_quad_tf_stamped,
            "level_quad",
            "bottom_camera_optical",
            time,
            timeout);

    if (!success) {
        return false;
    }

    geometry_msgs::PointStamped camera_position;
    tf2::doTransform(camera_position,
                     camera_position,
                     filtered_position_transform_stamped);
    current_altitude_ = camera_position.point.z;

    tf2::convert(filtered_position_transform_stamped.transform.rotation,
                 current_orientation_);

    current_camera_to_level_quad_tf_ = camera_to_level_quad_tf_stamped;
    return true;
}

} // namespace iarc7_vision
