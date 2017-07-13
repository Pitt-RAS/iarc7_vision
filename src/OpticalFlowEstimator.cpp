#include "iarc7_vision/OpticalFlowEstimator.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros_utils/SafeTransformWrapper.hpp>

#include <geometry_msgs/Vector3Stamped.h>
#include <iarc7_msgs/Float64Stamped.h>
#include <visualization_msgs/Marker.h>

namespace iarc7_vision {

OpticalFlowEstimator::OpticalFlowEstimator(
        const OpticalFlowEstimatorSettings& flow_estimator_settings,
        const OpticalFlowDebugSettings& debug_settings)
    : flow_estimator_settings_(flow_estimator_settings),
      debug_settings_(debug_settings),
      last_filtered_position_(),
      transform_wrapper_()
{
    ros::NodeHandle local_nh ("optical_flow_estimator");

    /*if (debug_settings_.debug_direction) {
        debug_direction_marker_pub_
            = local_nh.advertise<visualization_msgs::Marker>("direction", 1);
    }

    if (debug_settings_.debug_edges) {
        debug_edges_pub_ = local_nh.advertise<sensor_msgs::Image>("edges", 10);
    }

    if (debug_settings_.debug_lines) {
        debug_lines_pub_ = local_nh.advertise<sensor_msgs::Image>("lines", 10);
    }

    if (debug_settings_.debug_line_markers) {
        debug_line_markers_pub_ = local_nh.advertise<visualization_msgs::Marker>(
                "line_markers", 1);
    }*/

    twist_pub_
        = local_nh.advertise<geometry_msgs::TwistWithCovarianceStamped>("twist",
                                                                       10);

}

void OpticalFlowEstimator::update(const cv::Mat& image, const ros::Time&)
{
    if (last_filtered_position_.point.z
            >= flow_estimator_settings_.min_estimation_altitude) {
        try {
            geometry_msgs::TwistWithCovarianceStamped velocity;
            estimateVelocity(velocity, image, last_filtered_position_.point.z);
        } catch (const std::exception& ex) {
            ROS_ERROR_STREAM("Caught exception processing image: "
                          << ex.what());
        }
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

void OpticalFlowEstimator::estimateVelocity(geometry_msgs::TwistWithCovarianceStamped&,
                                         const cv::Mat&,
                                         double) const
{
    // m/px = camera_height / focal_length;
    //double current_meters_per_px = height
    //                     / getFocalLength(image.size(),
    //                                      flow_estimator_settings_.fov);

    // desired m_px used to keep kernel sizes relative to our features
    //double desired_meters_per_px = 1.0
    //                             / flow_estimator_settings_.pixels_per_meter;

    //double scale_factor = current_meters_per_px / desired_meters_per_px;

    //cv::Mat image_edges;
    if (cv::gpu::getCudaEnabledDeviceCount() == 0) {
        ROS_ERROR_ONCE("Optical Flow Estimator does not have a CPU version");

    } else {

    }
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

        /*current_filtered_position_(0) = camera_position.point.x;
        current_filtered_position_(1) = camera_position.point.y;
        current_filtered_position_(2) = std::isfinite(debug_settings_.debug_height)
                                      ? debug_settings_.debug_height
                                      : camera_position.point.z;*/
        //current_filtered_position_stamp_ = time;
    }
}

} // namespace iarc7_vision
