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
#include <sensor_msgs/Image.h>

#include "iarc7_safety/SafetyClient.hpp"
#include <iarc7_vision/VisionNodeConfig.h>
#include <ros_utils/ParamUtils.hpp>

#include "iarc7_vision/ColorCorrectionModel.hpp"
#include "iarc7_vision/GridLineEstimator.hpp"
#include "iarc7_vision/OpticalFlowEstimator.hpp"
#include "iarc7_vision/RoombaEstimator.hpp"
#include "iarc7_vision/RoombaImageLocation.hpp"
#include "iarc7_vision/UndistortionModel.hpp"

void getLineExtractorSettings(const ros::NodeHandle& private_nh,
                              iarc7_vision::LineExtractorSettings& line_settings)
{
    // Begin line extractor settings
    ROS_ASSERT(private_nh.getParam(
            "line_extractor/pixels_per_meter",
            line_settings.pixels_per_meter));

    ROS_ASSERT(private_nh.getParam(
            "line_extractor/canny_high_threshold",
            line_settings.canny_high_threshold));

    double canny_threshold_ratio;
    ROS_ASSERT(private_nh.getParam(
            "line_extractor/canny_threshold_ratio",
            canny_threshold_ratio));
    line_settings.canny_low_threshold =
        line_settings.canny_high_threshold / canny_threshold_ratio;

    ROS_ASSERT(private_nh.getParam(
            "line_extractor/canny_sobel_size",
            line_settings.canny_sobel_size));

    ROS_ASSERT(private_nh.getParam(
            "line_extractor/hough_rho_resolution",
            line_settings.hough_rho_resolution));

    ROS_ASSERT(private_nh.getParam(
            "line_extractor/hough_theta_resolution",
            line_settings.hough_theta_resolution));

    ROS_ASSERT(private_nh.getParam(
            "line_extractor/hough_thresh_fraction",
            line_settings.hough_thresh_fraction));

    ROS_ASSERT(private_nh.getParam(
            "line_extractor/fov",
            line_settings.fov));
}

void getOpticalFlowEstimatorSettings(const ros::NodeHandle& private_nh,
                    iarc7_vision::OpticalFlowEstimatorSettings& flow_settings)
{
    // Begin optical flow estimator settings
    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/fov",
            flow_settings.fov));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/min_estimation_altitude",
            flow_settings.min_estimation_altitude));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/camera_vertical_threshold",
            flow_settings.camera_vertical_threshold));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/points",
            flow_settings.points));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/quality_level",
            flow_settings.quality_level));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/min_dist",
            flow_settings.min_dist));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/win_size",
            flow_settings.win_size));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/max_level",
            flow_settings.max_level));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/iters",
            flow_settings.iters));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/scale_factor",
            flow_settings.scale_factor));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/crop",
            flow_settings.crop));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/crop_width",
            flow_settings.crop_width));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/crop_height",
            flow_settings.crop_height));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/variance",
            flow_settings.variance));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/variance_scale",
            flow_settings.variance_scale));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/x_cutoff_region_velocity_measurement",
            flow_settings.x_cutoff_region_velocity_measurement));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/y_cutoff_region_velocity_measurement",
            flow_settings.y_cutoff_region_velocity_measurement));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/debug_frameskip",
            flow_settings.debug_frameskip));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/tf_timeout",
            flow_settings.tf_timeout));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/max_rotational_vel",
            flow_settings.max_rotational_vel));

    std::string vector_filter_string;
    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/vector_filter",
            vector_filter_string));
    flow_settings.set_vector_filter_str(vector_filter_string);

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/min_vectors",
            flow_settings.min_vectors));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/max_filtered_variance",
            flow_settings.max_filtered_variance));

    ROS_ASSERT(private_nh.getParam(
            "optical_flow_estimator/max_normalized_element_variance",
            flow_settings.max_normalized_element_variance));

    ROS_ASSERT(private_nh.getParam(
        "optical_flow_estimator/hist_scale_factor",
        flow_settings.hist_scale_factor));

    ROS_ASSERT(private_nh.getParam(
        "optical_flow_estimator/hist_image_size_scale",
        flow_settings.hist_image_size_scale));
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
                        const ros::NodeHandle& private_nh,
                        iarc7_vision::LineExtractorSettings& line_settings,
                        iarc7_vision::OpticalFlowEstimatorSettings& flow_settings,
                        bool& ran)
{
    if (!ran) {
        getLineExtractorSettings(private_nh, line_settings);
        getOpticalFlowEstimatorSettings(private_nh, flow_settings);

        // Overwrite line extractor settings from dynamic reconfigure
        // with the ones from the param server
        config.pixels_per_meter       = line_settings.pixels_per_meter;
        config.canny_high_threshold   = line_settings.canny_high_threshold;
        config.canny_threshold_ratio  = line_settings.canny_high_threshold
                                            / line_settings.canny_low_threshold;
        config.canny_sobel_size       = line_settings.canny_sobel_size;
        config.hough_rho_resolution   = line_settings.hough_rho_resolution;
        config.hough_theta_resolution = line_settings.hough_theta_resolution;
        config.hough_thresh_fraction  = line_settings.hough_thresh_fraction;
        config.fov                    = line_settings.fov;

        // Overwrite optical flow settings from dynamic reconfigure
        // with the ones from the param server
        config.flow_fov             = flow_settings.fov;

        config.flow_min_estimation_altitude
            = flow_settings.min_estimation_altitude;

        config.flow_camera_vertical_threshold
            = flow_settings.camera_vertical_threshold;

        config.flow_points          = flow_settings.points;
        config.flow_quality_level   = flow_settings.quality_level;
        config.flow_min_dist        = flow_settings.min_dist;
        config.flow_win_size        = flow_settings.win_size;
        config.flow_iters           = flow_settings.iters;
        config.flow_max_level       = flow_settings.max_level;
        config.flow_scale_factor    = flow_settings.scale_factor;
        config.flow_variance        = flow_settings.variance;
        config.flow_variance_scale  = flow_settings.variance_scale;

        config.flow_x_cutoff_region_velocity_measurement =
            flow_settings.x_cutoff_region_velocity_measurement;

        config.flow_y_cutoff_region_velocity_measurement =
            flow_settings.y_cutoff_region_velocity_measurement;

        config.flow_debug_frameskip = flow_settings.debug_frameskip;
        config.flow_tf_timeout      = flow_settings.tf_timeout;

        config.flow_max_rotational_vel = flow_settings.max_rotational_vel;

        flow_settings.get_vector_filter_str(config.flow_vector_filter);

        config.flow_min_vectors = flow_settings.min_vectors;
        config.flow_max_filtered_variance = flow_settings.max_filtered_variance;
        config.flow_max_normalized_element_variance = flow_settings.max_normalized_element_variance;
        config.flow_hist_scale_factor = flow_settings.hist_scale_factor;
        config.flow_hist_image_size_scale = flow_settings.hist_image_size_scale;

        ran = true;
    } else {
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
        flow_settings.set_vector_filter_str(config.flow_vector_filter);
        flow_settings.min_vectors = config.flow_min_vectors;
        flow_settings.max_filtered_variance = config.flow_max_filtered_variance;
        flow_settings.max_normalized_element_variance = config.flow_max_normalized_element_variance;
        flow_settings.hist_scale_factor = config.flow_hist_scale_factor;
        flow_settings.hist_image_size_scale = config.flow_hist_image_size_scale;
    }
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

    int color_conversion_code;
    if (expected_image_format == "RGB") {
        color_conversion_code = 0;
    } else if (expected_image_format == "RGBA") {
        color_conversion_code = CV_RGBA2RGB;
    } else if (expected_image_format == "BGR") {
        color_conversion_code = CV_BGR2RGB;
    } else {
        ROS_ERROR("Invalid color conversion code");
        return 1;
    }

    // Create settings objects
    iarc7_vision::LineExtractorSettings line_extractor_settings;
    getLineExtractorSettings(private_nh, line_extractor_settings);
    iarc7_vision::GridEstimatorSettings grid_estimator_settings;
    getGridEstimatorSettings(private_nh, grid_estimator_settings);
    iarc7_vision::GridLineDebugSettings grid_line_debug_settings;
    getGridDebugSettings(private_nh, grid_line_debug_settings);

    iarc7_vision::OpticalFlowEstimatorSettings optical_flow_estimator_settings;
    getOpticalFlowEstimatorSettings(private_nh, optical_flow_estimator_settings);
    iarc7_vision::OpticalFlowDebugSettings optical_flow_debug_settings;

    // Load settings not in dynamic reconfigure
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
                               private_nh,
                               line_extractor_settings,
                               optical_flow_estimator_settings,
                               dynamic_reconfigure_called);
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
            "RGB"));
    optical_flow_estimator.reset(new iarc7_vision::OpticalFlowEstimator(
            optical_flow_estimator_settings,
            optical_flow_debug_settings,
            "RGB"));

    // Load the parameters specific to the vision node
    double startup_timeout;
    ROS_ASSERT(private_nh.getParam("startup_timeout", startup_timeout));

    size_t message_queue_item_limit = ros_utils::ParamUtils::getParam<int>(
            private_nh, "message_queue_item_limit");

    // Queue and callback for collecting images
    std::deque<sensor_msgs::Image::ConstPtr> message_queue;
    std::function<void(const sensor_msgs::Image::ConstPtr&)> image_msg_handler =
        [&](const sensor_msgs::Image::ConstPtr& message) {
            message_queue.push_back(message);

            // Make sure this doesn't grow without bound, but this will still trigger
            // the over-limit check in the main loop
            while (message_queue.size() > message_queue_item_limit) {
                message_queue.pop_front();
            }
        };

    image_transport::ImageTransport image_transporter{nh};
    image_transport::Subscriber sub = image_transporter.subscribe(
        "/bottom_image_raw/image_raw",
        100,
        image_msg_handler);

    ros::Publisher corrected_image_pub = nh.advertise<sensor_msgs::Image>("corrected_image", 1);

    // Loop rate
    ros::Rate rate (100);

    // Initialize the vision classes
    ros::Time start_time = ros::Time::now();
    ROS_ASSERT(gridline_estimator->waitUntilReady(ros::Duration(startup_timeout)));
    ROS_ASSERT(optical_flow_estimator->waitUntilReady(ros::Duration(startup_timeout)));
    while (message_queue.empty()) {
        if (!ros::ok()) {
            return 1;
        }

        if (ros::Time::now() > start_time + ros::Duration(startup_timeout)) {
            ROS_ERROR("Vision node timed out on startup waiting on images");
            return 1;
        }

        ros::spinOnce();
        rate.sleep();
    }

    const iarc7_vision::UndistortionModel undistortion_model(
            ros::NodeHandle("~/distortion_model"),
            cv::Size(message_queue.front()->width,
                     message_queue.front()->height));
    const iarc7_vision::ColorCorrectionModel color_correction_model(
            ros::NodeHandle("~/color_correction_model"));

    // Form a connection with the node monitor. If no connection can be made
    // assert because we don't know what's going on with the other nodes.
    ROS_INFO("vision_node: Attempting to form safety bond");
    Iarc7Safety::SafetyClient safety_client(nh, "vision_node");
    ROS_ASSERT_MSG(safety_client.formBond(),
                   "vision_node: Could not form bond with safety client");

    message_queue.clear();

    bool images_skipped = false;
    // Main loop
    while (ros::ok())
    {
        if (!message_queue.empty() && ros::ok()) {
            if (message_queue.size() > message_queue_item_limit - 1) {
                ROS_ERROR(
                        "Image queue has too many messages, clearing: %lu images",
                        message_queue.size());
                message_queue.clear();
                images_skipped = true;
                continue;
            }

            sensor_msgs::Image::ConstPtr message = message_queue.front();
            message_queue.pop_front();

            auto cv_shared_ptr = cv_bridge::toCvShare(message);
            cv::cuda::Stream cuda_stream = cv::cuda::Stream::Null();

            const auto start = std::chrono::high_resolution_clock::now();
            cv::cuda::GpuMat image_distorted;
            image_distorted.upload(cv_shared_ptr->image, cuda_stream);

            cv::cuda::GpuMat image_undistorted;
            undistortion_model.undistort(image_distorted,
                                         image_undistorted,
                                         cuda_stream);
            const auto distortion_time = std::chrono::high_resolution_clock::now();

            cv::cuda::GpuMat image_undistorted_rgb;
            if (color_conversion_code != 0) {
                cv::cuda::cvtColor(image_undistorted,
                                   image_undistorted_rgb,
                                   color_conversion_code,
                                   0,
                                   cuda_stream);
            } else {
                image_undistorted_rgb = image_undistorted;
            }

            const auto color_time = std::chrono::high_resolution_clock::now();

            cv::cuda::GpuMat image_correct;
            color_correction_model.correct(image_undistorted_rgb,
                                           image_correct,
                                           cuda_stream);

            {
                cv::Mat corrected_image_cpu;
                image_correct.download(corrected_image_cpu, cuda_stream);
                cuda_stream.waitForCompletion();

                cv_bridge::CvImage cv_image {
                    std_msgs::Header(),
                    sensor_msgs::image_encodings::RGB8,
                    corrected_image_cpu
                };

                corrected_image_pub.publish(cv_image.toImageMsg());
            }

            const auto color_correct_time = std::chrono::high_resolution_clock::now();

            //gridline_estimator->update(image_correct, message->header.stamp);
            const auto grid_time = std::chrono::high_resolution_clock::now();

            std::vector<iarc7_vision::RoombaImageLocation>
                                                      roomba_image_locations;
            roomba_estimator.update(image_correct,
                                    message->header.stamp,
                                    roomba_image_locations);
            const auto roomba_time = std::chrono::high_resolution_clock::now();

            optical_flow_estimator->update(image_correct,
                                           message->header.stamp,
                                           roomba_image_locations,
                                           images_skipped);
            const auto flow_time = std::chrono::high_resolution_clock::now();

            const auto count = [](const auto& a, const auto& b) {
                return std::chrono::duration_cast<std::chrono::microseconds>(
                        b - a).count();
            };

            ROS_DEBUG_STREAM(
                    "Distort: " << count(start, distortion_time) << std::endl
                 << "Color: " << count(distortion_time, color_time) << std::endl
                 << "Convert: " << count(color_time, color_correct_time) << std::endl
                 << "Grid: " << count(color_correct_time, grid_time) << std::endl
                 << "Roombas: " << count(grid_time, roomba_time) << std::endl
                 << "Optical Flow: " << count(roomba_time, flow_time));

            images_skipped = false;
        }
        else {
            rate.sleep();
        }

        ros::spinOnce();
    }

    // All is good.
    return 0;
}
