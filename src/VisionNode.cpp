// BAD HEADER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
#pragma GCC diagnostic pop
// END BAD HEADER

#include <image_transport/image_transport.h>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>

#include <dynamic_reconfigure/server.h>
#include <iarc7_vision/LineDetectorConfig.h>

#include <sensor_msgs/image_encodings.h>

#include "iarc7_vision/GridLineEstimator.hpp"

void getLineExtractorSettings(iarc7_vision::LineDetectorConfig &config,
                              const ros::NodeHandle& private_nh,
                              iarc7_vision::LineExtractorSettings& settings)
{
    static bool first_run = true;
    if (first_run) {
        ROS_ASSERT(private_nh.getParam(
                "line_extractor/pixels_per_meter",
                settings.pixels_per_meter));
        config.pixels_per_meter = settings.pixels_per_meter;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/canny_high_threshold",
                settings.canny_high_threshold));
        config.canny_high_threshold = settings.canny_high_threshold;

        double canny_threshold_ratio;
        ROS_ASSERT(private_nh.getParam(
                "line_extractor/canny_threshold_ratio",
                canny_threshold_ratio));
        settings.canny_low_threshold =
            settings.canny_high_threshold / canny_threshold_ratio;
        config.canny_threshold_ratio = canny_threshold_ratio;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/canny_sobel_size",
                settings.canny_sobel_size));
        config.canny_sobel_size = settings.canny_sobel_size;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/hough_rho_resolution",
                settings.hough_rho_resolution));
        config.hough_rho_resolution = settings.hough_rho_resolution;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/hough_theta_resolution",
                settings.hough_theta_resolution));
        config.hough_theta_resolution = settings.hough_theta_resolution;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/hough_thresh_fraction",
                settings.hough_thresh_fraction));
        config.hough_thresh_fraction = settings.hough_thresh_fraction;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/fov",
                settings.fov));
        config.fov = settings.fov;

        first_run = false;
    }
    else {
        settings.pixels_per_meter = config.pixels_per_meter;
        settings.canny_high_threshold = config.canny_high_threshold;

        settings.canny_low_threshold =
            config.canny_high_threshold / config.canny_threshold_ratio;

        settings.canny_sobel_size = config.canny_sobel_size;
        settings.hough_rho_resolution = config.hough_rho_resolution;
        settings.hough_theta_resolution = config.hough_theta_resolution;
        settings.hough_thresh_fraction = config.hough_thresh_fraction;
        settings.fov = config.fov;
    }
}

void getGridEstimatorSettings(const ros::NodeHandle& private_nh,
                              iarc7_vision::GridEstimatorSettings& settings)
{
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/theta_step",
            settings.theta_step));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/grid_step",
            settings.grid_step));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/grid_spacing",
            settings.grid_spacing));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/grid_line_thickness",
            settings.grid_line_thickness));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/grid_zero_offset_x",
            settings.grid_zero_offset(0)));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/grid_zero_offset_y",
            settings.grid_zero_offset(1)));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/grid_translation_mean_iterations",
            settings.grid_translation_mean_iterations));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/line_rejection_angle_threshold",
            settings.line_rejection_angle_threshold));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/min_extraction_altitude",
            settings.min_extraction_altitude));
    ROS_ASSERT(private_nh.getParam(
            "grid_estimator/allowed_position_stamp_error",
            settings.allowed_position_stamp_error));
}

void getDebugSettings(const ros::NodeHandle& private_nh,
                      iarc7_vision::GridLineDebugSettings& settings)
{
   ROS_ASSERT(private_nh.getParam(
            "grid_line_estimator/debug_line_detector",
            settings.debug_line_detector));
    ROS_ASSERT(private_nh.getParam(
            "grid_line_estimator/debug_direction",
            settings.debug_direction));
    ROS_ASSERT(private_nh.getParam(
            "grid_line_estimator/debug_edges",
            settings.debug_edges));
    ROS_ASSERT(private_nh.getParam(
            "grid_line_estimator/debug_lines",
            settings.debug_lines));
    ROS_ASSERT(private_nh.getParam(
            "grid_line_estimator/debug_line_markers",
            settings.debug_line_markers));
    if (private_nh.hasParam("grid_line_estimator/debug_height")) {
        ROS_ASSERT(private_nh.getParam(
            "grid_line_estimator/debug_height",
            settings.debug_height));
    } else {
        settings.debug_height = std::numeric_limits<double>::quiet_NaN();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vision");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    dynamic_reconfigure::Server<iarc7_vision::LineDetectorConfig> server;
    dynamic_reconfigure::Server
          <iarc7_vision::LineDetectorConfig>::CallbackType settings_callback;

    iarc7_vision::LineExtractorSettings line_extractor_settings;

    boost::function<void(iarc7_vision::LineDetectorConfig &config,
                         uint32_t level)> line_extractor_settings_callback =
        [&](iarc7_vision::LineDetectorConfig &config, uint32_t) {
            getLineExtractorSettings(config,
                                     private_nh,
                                     line_extractor_settings);
        };

    server.setCallback(line_extractor_settings_callback);

    iarc7_vision::GridEstimatorSettings grid_estimator_settings;
    getGridEstimatorSettings(private_nh, grid_estimator_settings);
    iarc7_vision::GridLineDebugSettings grid_line_debug_settings;
    getDebugSettings(private_nh, grid_line_debug_settings);
    iarc7_vision::GridLineEstimator gridline_estimator(
            line_extractor_settings,
            grid_estimator_settings,
            grid_line_debug_settings);

    ros::Rate rate (100);
    while (ros::ok() && ros::Time::now() == ros::Time(0)) {
        // wait
        ros::spinOnce();
    }

    std::vector<sensor_msgs::Image::ConstPtr> message_queue;

    std::function<void(const sensor_msgs::Image::ConstPtr&)> handler =
        [&](const sensor_msgs::Image::ConstPtr& message) {
            message_queue.push_back(message);
        };

    image_transport::ImageTransport image_transporter{nh};
    ros::Subscriber sub = nh.subscribe(
        "/bottom_image_raw/image_raw",
        100,
        &std::function<void(const sensor_msgs::Image::ConstPtr&)>::operator(),
        &handler);

    while (ros::ok())
    {
        if (message_queue.size() > 0) {
            const auto& message = message_queue.front();
            gridline_estimator.update(cv_bridge::toCvShare(message)->image,
                                      message->header.stamp);
            message_queue.erase(message_queue.begin());
        }

        ros::spinOnce();
        rate.sleep();
    }

    // All is good.
    return 0;
}
