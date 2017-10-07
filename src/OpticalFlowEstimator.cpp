#include "iarc7_vision/OpticalFlowEstimator.hpp"

// BAD HEADER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
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

#include <ros_utils/SafeTransformWrapper.hpp>

#include <geometry_msgs/PointStamped.h>
#include <iarc7_msgs/OrientationAnglesStamped.h>

namespace iarc7_vision {

static void download(const cv::gpu::GpuMat& d_mat,
                     std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const cv::gpu::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

static void drawArrows(cv::Mat& frame,
                       const std::vector<cv::Point2f>& prev_pts,
                       const std::vector<cv::Point2f>& next_pts,
                       const std::vector<uchar>& status,
                       cv::Scalar line_color = cv::Scalar(0, 0, 255))
{
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (status[i]) {
            int line_thickness = 1;

            cv::Point p = prev_pts[i];
            cv::Point q = next_pts[i];

            double angle = std::atan2(p.y - q.y, p.x - q.x);
            double hypotenuse = std::hypot(p.y - q.y, p.x - q.x);

            // Here we lengthen the arrow by a factor of three.
            q.x = p.x - 3 * hypotenuse * std::cos(angle);
            q.y = p.y - 3 * hypotenuse * std::sin(angle);

            // Now we draw the main line of the arrow.
            cv::line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = q.x + 9 * std::cos(angle + CV_PI / 4);
            p.y = q.y + 9 * std::sin(angle + CV_PI / 4);
            cv::line(frame, p, q, line_color, line_thickness);

            p.x = q.x + 9 * cos(angle - CV_PI / 4);
            p.y = q.y + 9 * sin(angle - CV_PI / 4);
            cv::line(frame, p, q, line_color, line_thickness);
        }
    }
}

OpticalFlowEstimator::OpticalFlowEstimator(
        const OpticalFlowEstimatorSettings& flow_estimator_settings,
        const OpticalFlowDebugSettings& debug_settings)
    : flow_estimator_settings_(flow_estimator_settings),
      debug_settings_(debug_settings),
      have_valid_last_image_(false),
      images_skipped_(0),
      last_scaled_image_(),
      last_scaled_grayscale_image_(),
      last_orientation_(),
      transform_wrapper_(),
      current_altitude_(0.0),
      current_orientation_(),
      last_message_time_(),
      expected_input_size_(),
      target_size_(),
      local_nh_("optical_flow_estimator"),
      debug_average_velocity_vector_image_pub_(
              local_nh_.advertise<sensor_msgs::Image>("average_vector_image",
                                                     1)),
      debug_velocity_vector_image_pub_(
              local_nh_.advertise<sensor_msgs::Image>("vector_image", 1)),
      correction_pub_(
              local_nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
                  "twist_correction", 10)),
      orientation_pub_(
              local_nh_.advertise<iarc7_msgs::OrientationAnglesStamped>(
                  "orientation", 10)),
      raw_pub_(
              local_nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
                  "twist_raw", 10)),
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
        cv::gpu::GpuMat scaled_image;
        cv::gpu::resize(last_scaled_image_,
                        scaled_image,
                        target_size_);
        last_scaled_image_ = scaled_image;

        cv::gpu::GpuMat scaled_grayscale_image;
        cv::gpu::cvtColor(last_scaled_image_,
                          scaled_grayscale_image,
                          CV_RGBA2GRAY);
        last_scaled_grayscale_image_ = scaled_grayscale_image;
    }

    return true;
}

void OpticalFlowEstimator::update(const sensor_msgs::Image::ConstPtr& message)
{
    // initial time for measurement
    const int64 start = cv::getTickCount();

    // make sure our current position is up to date
    if (!updateFilteredPosition(
                message->header.stamp,
                ros::Duration(flow_estimator_settings_.tf_timeout))) {
        ROS_ERROR("Unable to update position for optical flow");
        return;
    }

    if (debug_settings_.debug_times) {
        ROS_WARN("updateFilteredPosition: %f",
                 (cv::getTickCount() - start) / cv::getTickFrequency());
    }

    if (current_altitude_ < flow_estimator_settings_.min_estimation_altitude) {
        ROS_WARN("Height (%f) is below min processing height (%f)",
                 current_altitude_,
                 flow_estimator_settings_.min_estimation_altitude);
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
            const int64 start = cv::getTickCount();

            // Scale and convert input image
            cv::gpu::GpuMat d_frame1_big(curr_image);
            cv::gpu::GpuMat scaled_image;
            cv::gpu::GpuMat scaled_gray_image;

            cv::gpu::resize(d_frame1_big,
                            scaled_image,
                            target_size_);

            if (debug_settings_.debug_times) {
                ROS_WARN("post resize: %f",
                         (cv::getTickCount() - start) / cv::getTickFrequency());
            }

            cv::gpu::cvtColor(scaled_image,
                              scaled_gray_image,
                              CV_RGBA2GRAY);

            if (debug_settings_.debug_times) {
                ROS_WARN("post cvtColor: %f",
                         (cv::getTickCount() - start) / cv::getTickFrequency());
            }

            // Get velocity estimate from average vector
            processImage(scaled_image,
                         scaled_gray_image,
                         current_orientation_,
                         message->header.stamp,
                         current_altitude_,
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
}

bool OpticalFlowEstimator::waitUntilReady(
        const ros::Duration& startup_timeout)
{
    return updateFilteredPosition(ros::Time::now(), startup_timeout);
}

geometry_msgs::TwistWithCovarianceStamped
        OpticalFlowEstimator::estimateVelocity(
                const cv::Point2f& average_vec,
                const tf2::Quaternion& curr_orientation,
                const tf2::Quaternion& last_orientation,
                double height,
                const ros::Time& time) const
{
    // Get the pitch and roll of the camera in euler angles
    double yaw, pitch, roll;
    getYPR(curr_orientation, yaw, pitch, roll);

    double last_yaw, last_pitch, last_roll;
    getYPR(last_orientation, last_yaw, last_pitch, last_roll);

    iarc7_msgs::OrientationAnglesStamped ori_msg;
    ori_msg.header.stamp = time;
    ori_msg.data.pitch = pitch;
    ori_msg.data.roll = roll;
    ori_msg.data.yaw = yaw;
    orientation_pub_.publish(ori_msg);

    // m/px = camera_height / focal_length;
    // In the projected camera plane
    double current_meters_per_px = height
                         / getFocalLength(target_size_,
                                          flow_estimator_settings_.fov);

    // Calculate the average velocity
    cv::Point2f velocity_uncorrected;
    velocity_uncorrected.x = (average_vec.x * current_meters_per_px)
                           / std::cos(pitch)
                           / (time - last_message_time_).toSec();
    velocity_uncorrected.y = (average_vec.y * current_meters_per_px)
                           / -std::cos(roll)
                           / (time - last_message_time_).toSec();

    double dp;
    double dr;

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

    double angular_vel_y = dp / (time - last_message_time_).toSec();
    double angular_vel_x = dr / (time - last_message_time_).toSec();

    double distance_to_plane = current_altitude_
                             * std::sqrt(1.0
                                       + std::pow(std::tan(pitch), 2.0)
                                       + std::pow(std::tan(roll), 2.0));

    cv::Point2f correction_vel;
    correction_vel.x = -distance_to_plane
                     * angular_vel_y
                     / cos(pitch);
    correction_vel.y = distance_to_plane
                     * angular_vel_x
                     / cos(roll);

    cv::Point2f corrected_vel;
    corrected_vel.x = velocity_uncorrected.x - correction_vel.x;
    corrected_vel.y = velocity_uncorrected.y - correction_vel.y;

    last_pitch = pitch;
    last_roll = roll;

    // Fill out the twist
    geometry_msgs::TwistWithCovarianceStamped twist;
    twist.header.stamp = time;
    twist.header.frame_id = "level_quad";
    twist.twist.twist.linear.x = std::cos(yaw + M_PI) * -corrected_vel.x
                               - std::sin(yaw + M_PI) *  corrected_vel.y;
    twist.twist.twist.linear.y = std::cos(yaw + M_PI) *  corrected_vel.y
                               + std::sin(yaw + M_PI) * -corrected_vel.x;

    geometry_msgs::TwistWithCovarianceStamped twist_correction = twist;
    twist_correction.twist.twist.linear.x = correction_vel.x;
    twist_correction.twist.twist.linear.y = correction_vel.y;
    correction_pub_.publish(twist_correction);

    geometry_msgs::TwistWithCovarianceStamped twist_raw = twist;
    twist_raw.twist.twist.linear.x = velocity_uncorrected.x;
    twist_raw.twist.twist.linear.y = velocity_uncorrected.y;
    raw_pub_.publish(twist_raw);

    // Calculate variance
    double rotated_ang_vel_x = std::cos(yaw + M_PI) * angular_vel_x
                             - std::sin(yaw + M_PI) * angular_vel_y;
    double rotated_ang_vel_y = std::cos(yaw + M_PI) * angular_vel_y
                             + std::sin(yaw + M_PI) * angular_vel_x;
    double x_variance = std::pow(flow_estimator_settings_.variance_scale
                               * rotated_ang_vel_y, 2.0)
                      + flow_estimator_settings_.variance;
    double y_variance = std::pow(flow_estimator_settings_.variance_scale
                               * rotated_ang_vel_x, 2.0)
                      + flow_estimator_settings_.variance;

    twist.twist.covariance[0] = x_variance;
    twist.twist.covariance[7] = y_variance;

    return twist;
}

cv::Point2f OpticalFlowEstimator::findAverageVector(
        const std::vector<cv::Point2f>& tails,
        const std::vector<cv::Point2f>& heads,
        const std::vector<uchar>& status,
        const double x_cutoff,
        const double y_cutoff,
        const cv::Size& image_size)
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

            cv::Point p = tails[i];
            cv::Point q = heads[i];
            total_x += p.x - q.x;
            total_y += p.y - q.y;
            num_points++;
        }
    }

    cv::Point2f avg_vector;
    avg_vector.x = total_x / static_cast<double>(num_points);
    avg_vector.y = total_y / static_cast<double>(num_points);

    return avg_vector;
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
    int64 start = cv::getTickCount();

    // Create the feature detector and perform feature detection
    cv::gpu::GoodFeaturesToTrackDetector_GPU detector(
                    flow_estimator_settings_.points,
                    flow_estimator_settings_.quality_level,
                    flow_estimator_settings_.min_dist);
    cv::gpu::GpuMat d_prev_pts;

    detector(last_gray_frame, d_prev_pts);

    if (debug_settings_.debug_times) {
        ROS_WARN("post detector: %f",
                 (cv::getTickCount() - start) / cv::getTickFrequency());
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
        ROS_WARN("PYRLK sparse: %f",
                 (cv::getTickCount() - start) / cv::getTickFrequency());
    }

    tails.resize(d_prev_pts.cols);
    download(d_prev_pts, tails);

    heads.resize(d_next_pts.cols);
    download(d_next_pts, heads);

    status.resize(d_status.cols);
    download(d_status, status);

    // Publish debugging image with all vectors drawn
    if (debug && debug_settings_.debug_vectors_image) {
        // Draw arrows
        cv::Mat arrow_image;
        curr_frame.download(arrow_image);
        drawArrows(arrow_image,
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
                                        const tf2::Quaternion& orientation,
                                        const ros::Time& time,
                                        double height,
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

    // Calculate the average vector
    cv::Point2f average_vec = findAverageVector(
            tails,
            heads,
            status,
            flow_estimator_settings_.x_cutoff_region_velocity_measurement,
            flow_estimator_settings_.y_cutoff_region_velocity_measurement,
            target_size_);

    // Publish debugging image with average vector drawn
    if (debug && debug_settings_.debug_average_vector_image) {
        cv::Mat arrow_image;
        last_scaled_image_.download(arrow_image);

        const cv::Point2f start_point(target_size_.width / 2.0,
                                      target_size_.height / 2.0);

        const cv::Point2f end_point(average_vec.x + start_point.x,
                                    average_vec.y + start_point.y);

        drawArrows(arrow_image,
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

    const geometry_msgs::TwistWithCovarianceStamped velocity = estimateVelocity(
            average_vec,
            orientation,
            last_orientation_,
            height,
            time);

    // Publish velocity estimate
    twist_pub_.publish(velocity);
}

bool OpticalFlowEstimator::updateFilteredPosition(const ros::Time& time,
                                                  const ros::Duration& timeout)
{
    geometry_msgs::TransformStamped filtered_position_transform_stamped;
    if (!transform_wrapper_.getTransformAtTime(
            filtered_position_transform_stamped,
            "map",
            "bottom_camera_optical",
            time,
            timeout)) {
        ROS_ERROR("Failed to fetch transform to bottom_camera_optical");
        return false;
    } else {
        geometry_msgs::PointStamped camera_position;
        tf2::doTransform(camera_position,
                         camera_position,
                         filtered_position_transform_stamped);
        current_altitude_ = camera_position.point.z;

        tf2::convert(filtered_position_transform_stamped.transform.rotation,
                     current_orientation_);
        return true;
    }
}

} // namespace iarc7_vision
