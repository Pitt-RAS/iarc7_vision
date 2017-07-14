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
#include <iarc7_vision/VisionNodeConfig.h>
#include <iarc7_vision/OpticalFlowEstimatorConfig.h>

#include <sensor_msgs/image_encodings.h>

#include "iarc7_vision/GridLineEstimator.hpp"
#include "iarc7_vision/OpticalFlowEstimator.hpp"

void getDynamicSettings(iarc7_vision::VisionNodeConfig &config,
                        const ros::NodeHandle& private_nh,
                        iarc7_vision::LineExtractorSettings& line_settings,
                        iarc7_vision::OpticalFlowEstimatorSettings& flow_settings)
{
    static bool first_run = true;
    if (first_run) {
        // Begin line extractor settings
        ROS_ASSERT(private_nh.getParam(
                "line_extractor/pixels_per_meter",
                line_settings.pixels_per_meter));
        config.pixels_per_meter = line_settings.pixels_per_meter;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/canny_high_threshold",
                line_settings.canny_high_threshold));
        config.canny_high_threshold = line_settings.canny_high_threshold;

        double canny_threshold_ratio;
        ROS_ASSERT(private_nh.getParam(
                "line_extractor/canny_threshold_ratio",
                canny_threshold_ratio));
        line_settings.canny_low_threshold =
            line_settings.canny_high_threshold / canny_threshold_ratio;
        config.canny_threshold_ratio = canny_threshold_ratio;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/canny_sobel_size",
                line_settings.canny_sobel_size));
        config.canny_sobel_size = line_settings.canny_sobel_size;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/hough_rho_resolution",
                line_settings.hough_rho_resolution));
        config.hough_rho_resolution = line_settings.hough_rho_resolution;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/hough_theta_resolution",
                line_settings.hough_theta_resolution));
        config.hough_theta_resolution = line_settings.hough_theta_resolution;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/hough_thresh_fraction",
                line_settings.hough_thresh_fraction));
        config.hough_thresh_fraction = line_settings.hough_thresh_fraction;

        ROS_ASSERT(private_nh.getParam(
                "line_extractor/fov",
                line_settings.fov));
        config.fov = line_settings.fov;

        // Begin optical flow estimator settings
        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/pixels_per_meter",
                flow_settings.pixels_per_meter));
        config.flow_pixels_per_meter = flow_settings.pixels_per_meter;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/fov",
                flow_settings.fov));
        config.flow_fov = flow_settings.fov;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/min_estimation_altitude",
                flow_settings.min_estimation_altitude));
        config.flow_min_estimation_altitude = flow_settings.min_estimation_altitude;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/pixels_per_meter",
                flow_settings.pixels_per_meter));
        config.flow_pixels_per_meter = flow_settings.pixels_per_meter;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/points",
                flow_settings.points));
        config.flow_points = flow_settings.points;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/quality_level",
                flow_settings.quality_level));
        config.flow_quality_level = flow_settings.quality_level;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/min_dist",
                flow_settings.min_dist));
        config.flow_min_dist = flow_settings.min_dist;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/win_size",
                flow_settings.win_size));
        config.flow_min_dist = flow_settings.win_size;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/max_level",
                flow_settings.max_level));
        config.flow_max_level = flow_settings.max_level;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/iters",
                flow_settings.iters));
        config.flow_iters = flow_settings.iters;

        ROS_ASSERT(private_nh.getParam(
                "optical_flow_estimator/scale_factor",
                flow_settings.scale_factor));
        config.flow_scale_factor = flow_settings.scale_factor;

        first_run = false;
    }
    else {
        // Begin line extractor settings
        line_settings.pixels_per_meter = config.pixels_per_meter;
        line_settings.canny_high_threshold = config.canny_high_threshold;

        line_settings.canny_low_threshold =
            config.canny_high_threshold / config.canny_threshold_ratio;

        line_settings.canny_sobel_size = config.canny_sobel_size;
        line_settings.hough_rho_resolution = config.hough_rho_resolution;
        line_settings.hough_theta_resolution = config.hough_theta_resolution;
        line_settings.hough_thresh_fraction = config.hough_thresh_fraction;
        line_settings.fov = config.fov;

        // Begin optical flow estimator settings
        flow_settings.pixels_per_meter = config.flow_pixels_per_meter;
        flow_settings.fov = config.flow_fov;
        flow_settings.min_estimation_altitude = config.flow_min_estimation_altitude;
        flow_settings.points = config.flow_points;
        flow_settings.quality_level = config.flow_quality_level;
        flow_settings.min_dist = config.flow_min_dist;
        flow_settings.win_size = config.flow_win_size;
        flow_settings.max_level = config.flow_max_level;
        flow_settings.iters = config.flow_iters;
        flow_settings.scale_factor = config.flow_scale_factor;
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

void getGridDebugSettings(const ros::NodeHandle& private_nh,
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

void getFlowDebugSettings(const ros::NodeHandle& private_nh,
                          iarc7_vision::OpticalFlowDebugSettings& settings)
{
    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/debug_vectors_image",
            settings.debug_vectors_image));
    /*ROS_ASSERT(private_nh.getParam(
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
    }*/
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vision");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    dynamic_reconfigure::Server<iarc7_vision::VisionNodeConfig> dynamic_reconfigure_server;

    iarc7_vision::LineExtractorSettings line_extractor_settings;
    iarc7_vision::OpticalFlowEstimatorSettings optical_flow_estimator_settings;

    boost::function<void(iarc7_vision::VisionNodeConfig &config,
                         uint32_t level)> dynamic_reconfigure_settings_callback =
        [&](iarc7_vision::VisionNodeConfig &config, uint32_t) {
            getDynamicSettings(config,
                               private_nh,
                               line_extractor_settings,
                               optical_flow_estimator_settings);
        };

    dynamic_reconfigure_server.setCallback(dynamic_reconfigure_settings_callback);

    iarc7_vision::GridEstimatorSettings grid_estimator_settings;
    getGridEstimatorSettings(private_nh, grid_estimator_settings);
    iarc7_vision::GridLineDebugSettings grid_line_debug_settings;
    getGridDebugSettings(private_nh, grid_line_debug_settings);
    iarc7_vision::GridLineEstimator gridline_estimator(
            line_extractor_settings,
            grid_estimator_settings,
            grid_line_debug_settings);

    iarc7_vision::OpticalFlowDebugSettings optical_flow_debug_settings;
    getFlowDebugSettings(private_nh, optical_flow_debug_settings);
    iarc7_vision::OpticalFlowEstimator optical_flow_estimator(
            optical_flow_estimator_settings,
            optical_flow_debug_settings);

    ros::Rate rate (100);
    while (ros::ok() && ros::Time::now() == ros::Time(0)) {
        // wait
        ros::spinOnce();
    }

    double startup_timeout;
    ROS_ASSERT(private_nh.getParam("startup_timeout", startup_timeout));
    ROS_ASSERT(gridline_estimator.waitUntilReady(ros::Duration(startup_timeout)));

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

            const auto message = message_queue.front();
            message_queue.erase(message_queue.begin());
            gridline_estimator.update(cv_bridge::toCvShare(message)->image,
                                      message->header.stamp);

            optical_flow_estimator.update(message);
        }

        ros::spinOnce();
        rate.sleep();

    }

    // All is good.
    return 0;
}
