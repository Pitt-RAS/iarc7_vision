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
#include <ros_utils/SafeTransformWrapper.hpp>

#include <geometry_msgs/Vector3Stamped.h>
#include <iarc7_msgs/Float64Stamped.h>
#include <iarc7_msgs/OrientationAnglesStamped.h>
#include <visualization_msgs/Marker.h>

#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Vector3.h"

namespace iarc7_vision {

static void download(const cv::gpu::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
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

static void drawArrows(cv::Mat& frame, const std::vector<cv::Point2f>& prevPts, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status, cv::Scalar line_color = cv::Scalar(0, 0, 255))
{
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            int line_thickness = 1;

            cv::Point p = prevPts[i];
            cv::Point q = nextPts[i];

            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);

            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );

            if (hypotenuse < 1.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

OpticalFlowEstimator::OpticalFlowEstimator(
        ros::NodeHandle nh,
        const OpticalFlowEstimatorSettings& flow_estimator_settings,
        const OpticalFlowDebugSettings& debug_settings)
    : flow_estimator_settings_(flow_estimator_settings),
      debug_settings_(debug_settings),
      last_filtered_position_(),
      transform_wrapper_(),
      last_scaled_image_(),
      last_scaled_grayscale_image_(),
      debug_velocity_vector_image_pub_(),
      debug_average_velocity_vector_image_pub_(),
      imu_interpolator_(
        nh,
        "fc_imu",
        ros::Duration(flow_estimator_settings_.imu_update_timeout),
        ros::Duration(0),
        [](const sensor_msgs::Imu& msg) {
            return tf2::Vector3(msg.angular_velocity.x,
                                msg.angular_velocity.y,
                                msg.angular_velocity.z);
        },
        100),
      last_filtered_transform_stamped_(),
      last_message_time_(),
      correction_pub_(),
      raw_pub_()
    {
    ros::NodeHandle local_nh ("optical_flow_estimator");

    if (debug_settings_.debug_vectors_image) {
        debug_velocity_vector_image_pub_
            = local_nh.advertise<sensor_msgs::Image>("vector_image", 1);
    }

    if (debug_settings_.debug_average_vector_image) {
        debug_average_velocity_vector_image_pub_
            = local_nh.advertise<sensor_msgs::Image>("average_vector_image", 1);
    }

    twist_pub_
        = local_nh.advertise<geometry_msgs::TwistWithCovarianceStamped>("twist",
                                                                       10);

    ori_pub = local_nh.advertise<iarc7_msgs::OrientationAnglesStamped>("ori", 10);
    correction_pub_ = local_nh.advertise<geometry_msgs::TwistWithCovarianceStamped>("twist_correction", 10);
    raw_pub_ = local_nh.advertise<geometry_msgs::TwistWithCovarianceStamped>("twist_raw", 10);
}

bool OpticalFlowEstimator::waitUntilReady(const ros::Duration& startup_timeout) {
    bool success = imu_interpolator_.waitUntilReady(startup_timeout);
    if (!success) {
        ROS_ERROR("Failed to fetch initial fc imu message");
        return false;
    }

    updateFilteredPosition(ros::Time::now());

    return true;
}

void OpticalFlowEstimator::update(const sensor_msgs::Image::ConstPtr& message)
{
    int64 start = cv::getTickCount();
    updateFilteredPosition(message->header.stamp);
    ROS_WARN("updateFilteredPosition: %f", (cv::getTickCount() - start) / cv::getTickFrequency());

    if (last_filtered_position_.point.z
            >= flow_estimator_settings_.min_estimation_altitude) {
        static bool ran_once = false;
        if (ran_once) {
            try {
                // Variables needed for estimate velocity
                cv::Mat curr_image = cv_bridge::toCvShare(message)->image;
                geometry_msgs::TwistWithCovarianceStamped velocity;

                ROS_WARN("pre estimateVelocity: %f", (cv::getTickCount() - start) / cv::getTickFrequency());
                estimateVelocity(velocity, curr_image, last_filtered_position_.point.z, message->header.stamp);
                ROS_WARN("post estimateVelocity: %f", (cv::getTickCount() - start) / cv::getTickFrequency());

                twist_pub_.publish(velocity);

            } catch (const std::exception& ex) {
                ROS_ERROR_STREAM("Caught exception processing image flow: "
                              << ex.what());
            }
        }
        else {
            // TODO almost identical code is run when the scale is changed. Maybe make this a function?
            cv::Mat image = cv_bridge::toCvShare(message)->image;

            // Create the first last image
            cv::Size image_size = image.size();
            image_size.width = image_size.width * flow_estimator_settings_.scale_factor;
            image_size.height = image_size.height * flow_estimator_settings_.scale_factor;

            cv::gpu::GpuMat d_frame1_big(image);
            cv::gpu::GpuMat scaled_image;
            cv::gpu::GpuMat scaled_grayscale_image;

            ROS_WARN("w %d h %d", image_size.width, image_size.height);

            cv::gpu::resize(d_frame1_big,
                            scaled_image,
                            image_size);

            cv::gpu::cvtColor(scaled_image,
                              scaled_grayscale_image,
                              CV_RGBA2GRAY);

            last_scaled_image_ = scaled_image;
            last_scaled_grayscale_image_ = scaled_grayscale_image;

            ran_once = true;
        }

        // Always save off the last message time
        last_message_time_ = message->header.stamp;

    } else {
        ROS_WARN("Height (%f) is below min processing height (%f)",
                 last_filtered_position_.point.z,
                 flow_estimator_settings_.min_estimation_altitude);
    }
}

double OpticalFlowEstimator::getFocalLength(const cv::Size& img_size, double fov)
{
    return std::hypot(img_size.width/2.0, img_size.height/2.0)
         / std::tan(fov / 2.0);
}

void OpticalFlowEstimator::estimateVelocity(geometry_msgs::TwistWithCovarianceStamped& twist,
                                            const cv::Mat& image,
                                            double height,
                                            ros::Time time)
{
    cv::Size image_size = image.size();
    image_size.width = image_size.width * flow_estimator_settings_.scale_factor;
    image_size.height = image_size.height * flow_estimator_settings_.scale_factor;

    if (cv::gpu::getCudaEnabledDeviceCount() == 0) {
        ROS_ERROR_ONCE("Optical Flow Estimator does not have a CPU version");

    } else {

        static double last_scale = -1.0;
        // Fix scaling if the scaling changed
        if (last_scale != flow_estimator_settings_.scale_factor) {
            cv::gpu::GpuMat d_frame1_big(image);
            cv::gpu::GpuMat scaled_image;
            cv::gpu::GpuMat scaled_grayscale_image;

            cv::gpu::resize(d_frame1_big,
                            scaled_image,
                            image_size);

            cv::gpu::cvtColor(scaled_image,
                              scaled_grayscale_image,
                              CV_RGBA2GRAY);

            last_scaled_image_ = scaled_image;
            last_scaled_grayscale_image_ = scaled_grayscale_image;

            last_scale = flow_estimator_settings_.scale_factor;
        }

        int64 start = cv::getTickCount();

        // Scale and convert input image
        cv::gpu::GpuMat d_frame1_big(image);
        cv::gpu::GpuMat d_frame1;
        cv::gpu::GpuMat d_frame1Gray;

        cv::gpu::resize(d_frame1_big,
                        d_frame1,
                        image_size);
        ROS_WARN("post resize: %f", (cv::getTickCount() - start) / cv::getTickFrequency());

        cv::gpu::cvtColor(d_frame1,
                          d_frame1Gray,
                          CV_RGBA2GRAY);
        ROS_WARN("post cvtColor: %f", (cv::getTickCount() - start) / cv::getTickFrequency());

        // Create the feature detector and perform feature detection
        cv::gpu::GoodFeaturesToTrackDetector_GPU detector(
                        flow_estimator_settings_.points,
                        flow_estimator_settings_.quality_level,
                        flow_estimator_settings_.min_dist);
        cv::gpu::GpuMat d_prevPts;

        detector(last_scaled_grayscale_image_, d_prevPts);
        ROS_WARN("post detector: %f", (cv::getTickCount() - start) / cv::getTickFrequency());

        // Create optical flow object
        cv::gpu::PyrLKOpticalFlow d_pyrLK;

        d_pyrLK.winSize.width = flow_estimator_settings_.win_size;
        d_pyrLK.winSize.height = flow_estimator_settings_.win_size;
        d_pyrLK.maxLevel = flow_estimator_settings_.max_level;
        d_pyrLK.iters = flow_estimator_settings_.iters;

        cv::gpu::GpuMat d_nextPts;
        cv::gpu::GpuMat d_status;

        d_pyrLK.sparse(last_scaled_image_, d_frame1, d_prevPts, d_nextPts, d_status);
        ROS_WARN("PYRLK sparse: %f", (cv::getTickCount() - start) / cv::getTickFrequency());

        // Save off the current image
        last_scaled_image_ = d_frame1;
        last_scaled_grayscale_image_ = d_frame1Gray;

        // Calculate the average vector
        std::vector<cv::Point2f> prevPts(d_prevPts.cols);
        download(d_prevPts, prevPts);

        std::vector<cv::Point2f> nextPts(d_nextPts.cols);
        download(d_nextPts, nextPts);

        std::vector<uchar> status(d_status.cols);
        download(d_status, status);

        cv::Point2f average_vec = findAverageVector(prevPts,
                                                    nextPts,
                                                    status,
                                                    flow_estimator_settings_.x_cutoff_region_velocity_measurement,
                                                    flow_estimator_settings_.y_cutoff_region_velocity_measurement,
                                                    image_size);

        // Get the pitch and roll of the camera in eular angles
        tf2::Quaternion orientation;
        tf2::convert(last_filtered_transform_stamped_.transform.rotation, orientation);
        tf2::Matrix3x3 matrix;
        matrix.setRotation(orientation);
        double y, p, r;
        matrix.getEulerYPR(y, p, r);

        iarc7_msgs::OrientationAnglesStamped ori_msg;
        ori_msg.header.stamp = time;
        ori_msg.data.pitch = p;
        ori_msg.data.roll = r;
        ori_msg.data.yaw = y;
        ori_pub.publish(ori_msg);

        // m/px = camera_height / focal_length;
        // In the projected camera plane
        double current_meters_per_px = height
                             / getFocalLength(image_size,
                                              flow_estimator_settings_.fov);

        // Calculate the average velocity
        cv::Point2f velocity_uncorrected;
        velocity_uncorrected.x = (average_vec.x * current_meters_per_px) /
                                 std::cos(p) / 
                                 (time - last_message_time_).toSec();
        velocity_uncorrected.y = (average_vec.y * current_meters_per_px) /
                                 -std::cos(r) /
                                 (time - last_message_time_).toSec();


        double dp;
        double dr;

        if (last_p_ > CV_PI/2 && p < -CV_PI/2) dp = (p + 2*CV_PI - last_p_);
        else if (last_p_ < -CV_PI/2 && p > CV_PI/2) dp = (p - last_p_ - 2*CV_PI);
        else dp = (p - last_p_);

        if (last_r_ > CV_PI/2 && r < -CV_PI/2) dr = (r + 2*CV_PI - last_r_);
        else if (last_r_ < -CV_PI/2 && r > CV_PI/2) dr = (r - last_r_ - 2*CV_PI);
        else dr = (r - last_r_);

        double angular_vel_y = dp / (time - last_message_time_).toSec();
        double angular_vel_x = dr / (time - last_message_time_).toSec();

        double distance_to_plane = last_filtered_position_.point.z *
                                   sqrt(1 + std::pow(tan(p), 2.0) + std::pow(tan(r), 2.0));

        cv::Point2f correction_vel;
        correction_vel.x = -distance_to_plane *
                           angular_vel_y * cos(p);
        correction_vel.y = distance_to_plane *
                           angular_vel_x * cos(r);

        double hack_factor = 1.5;
        correction_vel.x *= hack_factor;
        correction_vel.y *= hack_factor;

        cv::Point2f corrected_vel;
        corrected_vel.x = velocity_uncorrected.x - correction_vel.x;
        corrected_vel.y = velocity_uncorrected.y - correction_vel.y;

        last_p_ = p;
        last_r_ = r;

        // Fill out the twist
        twist.header.stamp = time;
        twist.header.frame_id = "level_quad";
        twist.twist.twist.linear.x = std::cos(y + M_PI) * -corrected_vel.x - std::sin(y + M_PI) * corrected_vel.y;
        twist.twist.twist.linear.y = std::cos(y + M_PI) * corrected_vel.y + std::sin(y + M_PI) * -corrected_vel.x;

        geometry_msgs::TwistWithCovarianceStamped twist_correction = twist;
        twist_correction.twist.twist.linear.x = correction_vel.x;
        twist_correction.twist.twist.linear.y = correction_vel.y;
        correction_pub_.publish(twist_correction);

        geometry_msgs::TwistWithCovarianceStamped twist_raw = twist;
        twist_raw.twist.twist.linear.x = velocity_uncorrected.x;
        twist_raw.twist.twist.linear.y = velocity_uncorrected.y;
        raw_pub_.publish(twist_raw);

        // Calculate variance
        double rotated_ang_vel_x = std::cos(y + M_PI) * angular_vel_x - std::sin(y + M_PI) * angular_vel_y;
        double rotated_ang_vel_y = std::cos(y + M_PI) * angular_vel_y + std::sin(y + M_PI) * angular_vel_x;
        double x_variance = std::pow(flow_estimator_settings_.variance_scale
                                   * rotated_ang_vel_y, 2.0)
                          + flow_estimator_settings_.variance;
        double y_variance = std::pow(flow_estimator_settings_.variance_scale
                                   * rotated_ang_vel_x, 2.0)
                          + flow_estimator_settings_.variance;

        twist.twist.covariance[0] = x_variance;
        twist.twist.covariance[7] = y_variance;

        static int images_skipped = 0;
        if(images_skipped == 0) {
            // Publish debugging image with all vectors drawn
            if (debug_settings_.debug_vectors_image) {
                // Draw arrows
                cv::Mat arrow_image;
                d_frame1.download(arrow_image);
                drawArrows(arrow_image, prevPts, nextPts, status, cv::Scalar(255, 0, 0));

                cv_bridge::CvImage cv_image {
                    std_msgs::Header(),
                    sensor_msgs::image_encodings::RGBA8,
                    arrow_image
                };

                debug_velocity_vector_image_pub_.publish(cv_image.toImageMsg());
            }

            // Publish debugging image with average vector drawn
            if (debug_settings_.debug_average_vector_image) {
                // Draw arrows
                cv::Mat arrow_image;
                d_frame1.download(arrow_image);

                std::vector<cv::Point2f> end_point(1);
                std::vector<cv::Point2f> start_point(1);

                int x_off = image_size.width/2;
                int y_off = image_size.height/2;
                start_point[0].x = x_off;
                start_point[0].y = y_off;

                end_point[0].x = average_vec.x + x_off;
                end_point[0].y = average_vec.y + y_off;

                std::vector<uchar> fake_status(1);
                fake_status[0] = 1;
                drawArrows(arrow_image, start_point, end_point, fake_status, cv::Scalar(255, 0, 0));

                cv_bridge::CvImage cv_image {
                    std_msgs::Header(),
                    sensor_msgs::image_encodings::RGBA8,
                    arrow_image
                };

                debug_average_velocity_vector_image_pub_.publish(cv_image.toImageMsg());
            }
        }
        images_skipped = (images_skipped + 1) % flow_estimator_settings_.debug_frameskip;

    }
}

cv::Point2f OpticalFlowEstimator::findAverageVector(const std::vector<cv::Point2f>& prevPts,
                                                    const std::vector<cv::Point2f>& nextPts,
                                                    const std::vector<uchar>& status,
                                                    const double x_cutoff,
                                                    const double y_cutoff,
                                                    const cv::Size& image_size) {
    double averageX = 0.0;
    double averageY = 0.0;
    int num_points = 0;

    // TODO filter properly based on magnitude, angle and standard deviation
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i] &&
            prevPts[i].x > image_size.width * x_cutoff &&
            prevPts[i].x < image_size.width * (1.0 - x_cutoff) &&
            prevPts[i].y > image_size.height * y_cutoff &&
            prevPts[i].y < image_size.height * (1.0 - y_cutoff))
        {
            cv::Point p = prevPts[i];
            cv::Point q = nextPts[i];

            averageX += p.x - q.x;
            averageY += p.y - q.y;
            num_points++;
        }
    }

    cv::Point2f velocity_vector;
    velocity_vector.x = averageX / static_cast<double>(num_points);
    velocity_vector.y = averageY / static_cast<double>(num_points);

    return velocity_vector;
}

void OpticalFlowEstimator::updateFilteredPosition(const ros::Time& time)
{
    geometry_msgs::TransformStamped filtered_position_transform_stamped;
    if (!transform_wrapper_.getTransformAtTime(
            filtered_position_transform_stamped,
            "map",
            "bottom_camera_optical",
            time,
            ros::Duration(1.0))) {
        ROS_ERROR("Failed to fetch transform to bottom_camera_optical");
    } else {
        geometry_msgs::PointStamped camera_position;
        tf2::doTransform(camera_position,
                         camera_position,
                         filtered_position_transform_stamped);

        last_filtered_position_ = camera_position;
        last_filtered_transform_stamped_ = filtered_position_transform_stamped;
    }

    // Get the current acceleration of the quad
    // TODO transform this into the cameras frame
    bool success = imu_interpolator_.getInterpolatedMsgAtTime(last_angular_velocity_, time);
    if (!success) {
        ROS_ERROR("Failed to get imu information in OpticalFlowEstimator::updateFilteredPosition");
    }
}

} // namespace iarc7_vision
