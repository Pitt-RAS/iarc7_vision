// BAD HEADER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
#pragma GCC diagnostic pop
// END BAD HEADER

#include <chrono>
#include <deque>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#include "iarc7_safety/SafetyClient.hpp"
#include <iarc7_vision/VisionNodeConfig.h>
#include <ros_utils/ParamUtils.hpp>

#include "iarc7_vision/GridLineEstimator.hpp"
#include "iarc7_vision/OpticalFlowEstimator.hpp"
#include "iarc7_vision/RoombaEstimator.hpp"
#include "iarc7_vision/RoombaImageLocation.hpp"

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
        "optical_flow_estimator/debug_average_vector_image",
        settings.debug_average_vector_image));

    ROS_ASSERT(private_nh.getParam(
        "optical_flow_estimator/debug_intermediate_velocities",
        settings.debug_intermediate_velocities));

    ROS_ASSERT(private_nh.getParam(
        "optical_flow_estimator/debug_orientation",
        settings.debug_orientation));

    ROS_ASSERT(private_nh.getParam(
        "optical_flow_estimator/debug_times",
        settings.debug_times));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/debug_vectors_image",
            settings.debug_vectors_image));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/debug_hist",
            settings.debug_hist));
}

void getDynamicSettings(iarc7_vision::VisionNodeConfig &config,
                        iarc7_vision::LineExtractorSettings& line_settings,
                        iarc7_vision::OpticalFlowEstimatorSettings& flow_settings)
{
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
    flow_settings.fov = config.flow_fov;
    flow_settings.min_estimation_altitude
        = config.flow_min_estimation_altitude;
    flow_settings.camera_vertical_threshold
        = config.flow_camera_vertical_threshold;
    flow_settings.points = config.flow_points;
    flow_settings.quality_level = config.flow_quality_level;
    flow_settings.min_dist = config.flow_min_dist;
    flow_settings.win_size = config.flow_win_size;
    flow_settings.max_level = config.flow_max_level;
    flow_settings.iters = config.flow_iters;
    flow_settings.scale_factor = config.flow_scale_factor;
    flow_settings.variance = config.flow_variance;
    flow_settings.variance_scale = config.flow_variance_scale;
    flow_settings.x_cutoff_region_velocity_measurement =
        config.flow_x_cutoff_region_velocity_measurement;
    flow_settings.y_cutoff_region_velocity_measurement =
        config.flow_y_cutoff_region_velocity_measurement;
    flow_settings.debug_frameskip = config.flow_debug_frameskip;
    flow_settings.tf_timeout = config.flow_tf_timeout;
    flow_settings.max_rotational_vel = config.flow_max_rotational_vel;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vision");

    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
        ROS_ERROR("No CUDA devices found, Vision Node cannot run");
        return 1;
    }

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::string expected_image_format
        = ros_utils::ParamUtils::getParam<std::string>(
                private_nh, "image_format");

    // Create settings objects
    iarc7_vision::LineExtractorSettings line_extractor_settings;
    iarc7_vision::GridEstimatorSettings grid_estimator_settings;
    iarc7_vision::GridLineDebugSettings grid_line_debug_settings;
    iarc7_vision::OpticalFlowEstimatorSettings optical_flow_estimator_settings;
    iarc7_vision::OpticalFlowDebugSettings optical_flow_debug_settings;

    // Load settings not in dynamic reconfigure
    getGridEstimatorSettings(private_nh, grid_estimator_settings);
    getGridDebugSettings(private_nh, grid_line_debug_settings);
    getFlowDebugSettings(private_nh, optical_flow_debug_settings);

    std::unique_ptr<iarc7_vision::GridLineEstimator> gridline_estimator;
    std::unique_ptr<iarc7_vision::OpticalFlowEstimator> optical_flow_estimator;

    // Set up dynamic reconfigure
    dynamic_reconfigure::Server<iarc7_vision::VisionNodeConfig> dynamic_reconfigure_server;
    bool dynamic_reconfigure_called = false;
    boost::function<void(iarc7_vision::VisionNodeConfig &config,
                         uint32_t level)> dynamic_reconfigure_settings_callback =
        [&](iarc7_vision::VisionNodeConfig &config, uint32_t) {
            getDynamicSettings(config,
                               line_extractor_settings,
                               optical_flow_estimator_settings);
            dynamic_reconfigure_called = true;

            if (gridline_estimator != nullptr) {
                ROS_ASSERT(gridline_estimator->onSettingsChanged());
            }

            if (optical_flow_estimator != nullptr) {
                ROS_ASSERT(optical_flow_estimator->onSettingsChanged());
            }
        };
    dynamic_reconfigure_server.setCallback(
            dynamic_reconfigure_settings_callback);

    // Wait for time to begin and settings to be fetched
    while (ros::ok() &&
           (ros::Time::now() == ros::Time(0) ||
           !dynamic_reconfigure_called)) {
        // wait
        ros::spinOnce();
    }

    // Create vision processing objects
    iarc7_vision::RoombaEstimator roomba_estimator;
    gridline_estimator.reset(new iarc7_vision::GridLineEstimator(
            line_extractor_settings,
            grid_estimator_settings,
            grid_line_debug_settings,
            expected_image_format));
    optical_flow_estimator.reset(new iarc7_vision::OpticalFlowEstimator(
            optical_flow_estimator_settings,
            optical_flow_debug_settings,
            expected_image_format));

    // Queue and callback for collecting images
    std::deque<sensor_msgs::Image::ConstPtr> message_queue;
    std::function<void(const sensor_msgs::Image::ConstPtr&)> image_msg_handler =
        [&](const sensor_msgs::Image::ConstPtr& message) {
            message_queue.push_back(message);
        };

    image_transport::ImageTransport image_transporter{nh};
    image_transport::Subscriber sub = image_transporter.subscribe(
        "/bottom_image_raw/image_raw",
        100,
        image_msg_handler);

    // Load the parameters specific to the vision node
    double startup_timeout;
    ROS_ASSERT(private_nh.getParam("startup_timeout", startup_timeout));

    size_t message_queue_item_limit = ros_utils::ParamUtils::getParam<int>(
            private_nh, "message_queue_item_limit");

    // Loop rate
    ros::Rate rate (300);

    // Initialize the vision classes
    ros::Time start_time = ros::Time::now();
    ROS_ASSERT(gridline_estimator->waitUntilReady(ros::Duration(startup_timeout)));
    ROS_ASSERT(optical_flow_estimator->waitUntilReady(ros::Duration(startup_timeout)));
    while (message_queue.empty()) {
        if (!ros::ok()) {
            return 1;
        }

        if (ros::Time::now() > start_time + ros::Duration(startup_timeout)) {
            ROS_ERROR("Vision node timed out on startup");
            return 1;
        }

        rate.sleep();
    }

    // Form a connection with the node monitor. If no connection can be made
    // assert because we don't know what's going on with the other nodes.
    ROS_INFO("vision_node: Attempting to form safety bond");
    Iarc7Safety::SafetyClient safety_client(nh, "vision_node");
    ROS_ASSERT_MSG(safety_client.formBond(),
                   "vision_node: Could not form bond with safety client");

    message_queue.clear();

    // Main loop
    while (ros::ok())
    {
        if (!message_queue.empty()) {
            if (message_queue.size() > message_queue_item_limit - 1) {
                ROS_ERROR(
                        "Image queue has too many messages, clearing: %lu images",
                        message_queue.size());
                message_queue.clear();
                continue;
            }

            sensor_msgs::Image::ConstPtr message = message_queue.front();
            message_queue.pop_front();

            auto cv_shared_ptr = cv_bridge::toCvShare(message);
            cv::cuda::GpuMat image(cv_shared_ptr->image);

            const auto start = std::chrono::high_resolution_clock::now();
            gridline_estimator->update(image, message->header.stamp);
            const auto grid_time = std::chrono::high_resolution_clock::now();

            std::vector<iarc7_vision::RoombaImageLocation>
                                                      roomba_image_locations;
            roomba_estimator.update(image,
                                    message->header.stamp,
                                    roomba_image_locations);
            const auto roomba_time = std::chrono::high_resolution_clock::now();

            optical_flow_estimator->update(image,
                                           message->header.stamp,
                                           roomba_image_locations);
            const auto flow_time = std::chrono::high_resolution_clock::now();

            ROS_DEBUG_STREAM(
                    "Grid: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                        grid_time - start).count()
                    << " Flow: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                        flow_time - roomba_time).count()
                    << " Roomba: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                        roomba_time - grid_time).count());
        }

        ros::spinOnce();
        rate.sleep();
    }

    // All is good.
    return 0;
}
