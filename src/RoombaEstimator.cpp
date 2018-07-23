#include "iarc7_vision/RoombaEstimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <string>
#include <tf/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PointStamped.h>
#include <iarc7_msgs/RoombaDetection.h>
#include <iarc7_msgs/RoombaDetectionFrame.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include "iarc7_vision/cv_utils.hpp"
#include "iarc7_vision/RoombaEstimatorConfig.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Dense>
#pragma GCC diagnostic pop

namespace iarc7_vision
{

RoombaEstimator::RoombaEstimator(const cv::Size& image_size)
    : nh_(),
      private_nh_("~/roomba_estimator"),
      dynamic_reconfigure_server_(private_nh_),
      dynamic_reconfigure_settings_callback_(
              [this](iarc7_vision::RoombaEstimatorConfig& config, uint32_t) {
                  getDynamicSettings(config);
              }),
      dynamic_reconfigure_called_(false),
      transform_wrapper_(),
      camera_to_map_tf_(),
      roomba_pub_(nh_.advertise<iarc7_msgs::RoombaDetectionFrame>(
                  "detected_roombas", 100)),
      settings_(getSettings(private_nh_)),
      input_size_(image_size),
      detection_size_(settings_.detection_image_width,
                      input_size_.height
                    * settings_.detection_image_width / input_size_.width),
      blob_detector_(std::make_unique<const RoombaBlobDetector>(
                  settings_, private_nh_, detection_size_))
{
    dynamic_reconfigure_server_.setCallback(
            dynamic_reconfigure_settings_callback_);

    if (settings_.debug_detected_rects) {
        debug_detected_rects_pub_ = private_nh_.advertise<sensor_msgs::Image>(
                "detected_rects", 10);
    }
}

void RoombaEstimator::pixelToRay(double px,
                                 double py,
                                 double pw,
                                 double ph,
                                 geometry_msgs::Vector3Stamped& ray) const
{
    px -= pw * 0.5;
    py -= ph * 0.5;

    double pix_r = std::hypot(px, py);
    double pix_R = std::hypot(ph, pw) * 0.5;

    double max_phi = settings_.bottom_camera_aov * M_PI / 360;
    double pix_focal = pix_R / std::tan(max_phi);
    double theta = std::atan2(py, px);

    double camera_radius = pix_r / pix_focal;

    ray.vector.x = camera_radius * std::cos(theta);
    ray.vector.y = camera_radius * std::sin(theta);
    ray.vector.z = 1;

    double norm = std::sqrt(std::pow(ray.vector.x, 2)
                          + std::pow(ray.vector.y, 2)
                          + std::pow(ray.vector.z, 2));

    ray.vector.x /= norm;
    ray.vector.y /= norm;
    ray.vector.z /= norm;
}

void RoombaEstimator::getDynamicSettings(
        iarc7_vision::RoombaEstimatorConfig& config)
{
    if (!dynamic_reconfigure_called_) {
        config.detection_image_width = settings_.detection_image_width;

        config.hsv_slice_h_green_min = settings_.hsv_slice_h_green_min;
        config.hsv_slice_h_green_max = settings_.hsv_slice_h_green_max;
        config.hsv_slice_h_red1_min  = settings_.hsv_slice_h_red1_min;
        config.hsv_slice_h_red1_max  = settings_.hsv_slice_h_red1_max;
        config.hsv_slice_h_red2_min  = settings_.hsv_slice_h_red2_min;
        config.hsv_slice_h_red2_max  = settings_.hsv_slice_h_red2_max;
        config.hsv_slice_s_min       = settings_.hsv_slice_s_min;
        config.hsv_slice_s_max       = settings_.hsv_slice_s_max;
        config.hsv_slice_v_min       = settings_.hsv_slice_v_min;
        config.hsv_slice_v_max       = settings_.hsv_slice_v_max;

        config.min_roomba_blob_size  = settings_.min_roomba_blob_size;
        config.max_roomba_blob_size  = settings_.max_roomba_blob_size;

        config.morphology_size = settings_.morphology_size;
        config.morphology_iterations = settings_.morphology_iterations;

        dynamic_reconfigure_called_ = true;
    } else {
        settings_.detection_image_width = config.detection_image_width;

        settings_.hsv_slice_h_green_min = config.hsv_slice_h_green_min;
        settings_.hsv_slice_h_green_max = config.hsv_slice_h_green_max;
        settings_.hsv_slice_h_red1_min = config.hsv_slice_h_red1_min;
        settings_.hsv_slice_h_red1_max = config.hsv_slice_h_red1_max;
        settings_.hsv_slice_h_red2_min = config.hsv_slice_h_red2_min;
        settings_.hsv_slice_h_red2_max = config.hsv_slice_h_red2_max;
        settings_.hsv_slice_s_min = config.hsv_slice_s_min;
        settings_.hsv_slice_s_max = config.hsv_slice_s_max;
        settings_.hsv_slice_v_min = config.hsv_slice_v_min;
        settings_.hsv_slice_v_max = config.hsv_slice_v_max;

        settings_.min_roomba_blob_size = config.min_roomba_blob_size;
        settings_.max_roomba_blob_size = config.max_roomba_blob_size;

        settings_.morphology_size = config.morphology_size;
        settings_.morphology_iterations = config.morphology_iterations;

        detection_size_ = cv::Size(settings_.detection_image_width,
                        input_size_.height
                      * settings_.detection_image_width / input_size_.width);
        blob_detector_ = std::make_unique<const RoombaBlobDetector>(
                    settings_, private_nh_, detection_size_);
    }
}

RoombaEstimatorSettings RoombaEstimator::getSettings(
        const ros::NodeHandle& private_nh)
{

#define IARC7_VISION_RES_LOAD(x) \
    ROS_ASSERT(private_nh.getParam( \
               #x, \
               settings.x));

    RoombaEstimatorSettings settings;
    IARC7_VISION_RES_LOAD(roomba_plate_width);
    IARC7_VISION_RES_LOAD(roomba_plate_height);
    IARC7_VISION_RES_LOAD(roomba_height);
    IARC7_VISION_RES_LOAD(detection_image_width);
    IARC7_VISION_RES_LOAD(hsv_slice_h_green_min);
    IARC7_VISION_RES_LOAD(hsv_slice_h_green_max);
    IARC7_VISION_RES_LOAD(hsv_slice_h_red1_min);
    IARC7_VISION_RES_LOAD(hsv_slice_h_red1_max);
    IARC7_VISION_RES_LOAD(hsv_slice_h_red2_min);
    IARC7_VISION_RES_LOAD(hsv_slice_h_red2_max);
    IARC7_VISION_RES_LOAD(hsv_slice_s_min);
    IARC7_VISION_RES_LOAD(hsv_slice_s_max);
    IARC7_VISION_RES_LOAD(hsv_slice_v_min);
    IARC7_VISION_RES_LOAD(hsv_slice_v_max);
    IARC7_VISION_RES_LOAD(min_roomba_blob_size);
    IARC7_VISION_RES_LOAD(max_roomba_blob_size);
    IARC7_VISION_RES_LOAD(morphology_size);
    IARC7_VISION_RES_LOAD(morphology_iterations);
    IARC7_VISION_RES_LOAD(morphology_size);
    IARC7_VISION_RES_LOAD(uncertainty_scale);
    IARC7_VISION_RES_LOAD(bottom_camera_aov);
    IARC7_VISION_RES_LOAD(debug_hsv_slice);
    IARC7_VISION_RES_LOAD(debug_contours);
    IARC7_VISION_RES_LOAD(debug_detected_rects);

#undef IARC7_VISION_RES_LOAD

    return settings;
}

double RoombaEstimator::getHeight(const ros::Time& time)
{
    if (!transform_wrapper_.getTransformAtTime(
                camera_to_map_tf_,
                "map",
                "bottom_camera_rgb_optical_frame",
                time,
                ros::Duration(1.0))) {
        throw ros::Exception("Failed to fetch transform");
    }
    return camera_to_map_tf_.transform.translation.z;
}

void RoombaEstimator::calcBoxUncertainties(
       const cv::Size& image_size,
       const std::vector<cv::RotatedRect>& bounding_rects,
       std::vector<std::array<double, 4>>& position_covariances,
       std::vector<double>& box_uncertainties) const
{
    ROS_ASSERT(box_uncertainties.empty());
    box_uncertainties.reserve(bounding_rects.size());

    ROS_ASSERT(position_covariances.empty());
    position_covariances.reserve(bounding_rects.size());

    geometry_msgs::PointStamped cam_pos;
    tf2::doTransform(cam_pos, cam_pos, camera_to_map_tf_);

    for (size_t i = 0; i < bounding_rects.size(); i++) {
        const cv::RotatedRect& bounding_rect = bounding_rects[i];

        geometry_msgs::Vector3Stamped roomba_center_ray;
        pixelToRay(bounding_rect.center.x,
                   bounding_rect.center.y,
                   image_size.width,
                   image_size.height,
                   roomba_center_ray);

        tf2::doTransform(roomba_center_ray,
                         roomba_center_ray,
                         camera_to_map_tf_);

        const double roomba_center_ray_scale =
            (cam_pos.point.z - settings_.roomba_height)
          / -roomba_center_ray.vector.z;

        const double plate_location_x =
            roomba_center_ray.vector.x * roomba_center_ray_scale;
        const double plate_location_y =
            roomba_center_ray.vector.y * roomba_center_ray_scale;

        geometry_msgs::Vector3Stamped roomba_plate_side_ray;
        pixelToRay(bounding_rect.center.x + bounding_rect.size.width,
                   bounding_rect.center.y,
                   image_size.width,
                   image_size.height,
                   roomba_plate_side_ray);

        tf2::doTransform(roomba_plate_side_ray,
                         roomba_plate_side_ray,
                         camera_to_map_tf_);

        const double roomba_plate_side_ray_scale =
            (cam_pos.point.z - settings_.roomba_height)
          / -roomba_plate_side_ray.vector.z;

        const double plate_side_location_x =
            roomba_plate_side_ray.vector.x * roomba_plate_side_ray_scale;
        const double plate_side_location_y =
            roomba_plate_side_ray.vector.y * roomba_plate_side_ray_scale;

        const double meters_per_pixel = std::hypot(
                plate_location_x - plate_side_location_x,
                plate_location_y - plate_side_location_y)
            / bounding_rect.size.width;

        const double plate_width_meters = bounding_rect.size.width * meters_per_pixel;
        const double plate_height_meters = bounding_rect.size.height * meters_per_pixel;

        const double position_error = std::max(
                std::abs(plate_width_meters - settings_.roomba_plate_width),
                std::abs(plate_height_meters - settings_.roomba_plate_height));

        position_covariances.push_back({
                position_error * position_error,
                0,
                0,
                position_error * position_error});

        const double relative_error =
                std::abs(plate_width_meters - settings_.roomba_plate_width)
                            / settings_.roomba_plate_width
              + std::abs(plate_height_meters - settings_.roomba_plate_height)
                            / settings_.roomba_plate_height;

        const double uncertainty = settings_.uncertainty_scale * relative_error;

        box_uncertainties.push_back(uncertainty);
    }
}

void RoombaEstimator::calcFloorPoly(geometry_msgs::Polygon& poly)
{
    poly.points.clear();

    geometry_msgs::Vector3Stamped camera_ray;
    geometry_msgs::Point32 point;
    double scale;

    geometry_msgs::PointStamped camera_position;
    tf2::doTransform(camera_position, camera_position, camera_to_map_tf_);

    for (const auto& image_point : {
            std::make_pair(0, 0),
            std::make_pair(0, 1),
            std::make_pair(1, 1),
            std::make_pair(1, 0)}) {
        pixelToRay(image_point.first, image_point.second, 1, 1, camera_ray);
        tf2::doTransform(camera_ray, camera_ray, camera_to_map_tf_);
        scale = camera_position.point.z / -camera_ray.vector.z;
        point.x = camera_position.point.x + camera_ray.vector.x * scale;
        point.y = camera_position.point.y + camera_ray.vector.y * scale;
        poly.points.push_back(point);
    }
}

void RoombaEstimator::calcPose(const cv::Point2f& pos,
                               double angle,
                               double pw,
                               double ph,
                               iarc7_msgs::RoombaDetection& roomba,
                               RoombaImageLocation& roomba_image_location)
{
    geometry_msgs::PointStamped cam_pos;
    tf2::doTransform(cam_pos, cam_pos, camera_to_map_tf_);

    geometry_msgs::Vector3Stamped camera_ray;
    pixelToRay(pos.x, pos.y, pw, ph, camera_ray);

    geometry_msgs::Vector3Stamped map_ray;
    tf2::doTransform(camera_ray, map_ray, camera_to_map_tf_);

    double ray_scale = (cam_pos.point.z - settings_.roomba_height)
                     / -map_ray.vector.z;

    roomba.pose.x = cam_pos.point.x + map_ray.vector.x * ray_scale;
    roomba.pose.y = cam_pos.point.y + map_ray.vector.y * ray_scale;

    geometry_msgs::Vector3Stamped roomba_yaw;
    geometry_msgs::Vector3Stamped roomba_yaw_after;
    roomba_yaw.vector.x = std::cos(angle);
    roomba_yaw.vector.y = std::sin(angle);
    tf2::doTransform(roomba_yaw, roomba_yaw_after, camera_to_map_tf_);

    roomba.pose.theta = std::atan2(roomba_yaw_after.vector.y,
                                   roomba_yaw_after.vector.x);

    roomba_image_location.x = pos.x / pw;
    roomba_image_location.y = pos.y / pw;

    // ray_scale is equivalent to the distance of the roomba
    // from the camera center
    // Roomba radius is hard coded to 0.2m
    // It was adjusted to 0.3m because of errors with the below equations
    // The radius is a value from 0-1 which maps across the diagnol of the image
    double size_relative_diagonal = 0.3 
                                    / (2.0 * ray_scale 
                                       * std::tan(settings_.bottom_camera_aov
                                                  * M_PI / 180.0
                                                  / 2.0));

    // Calculate the ratio in terms of the image width
    double theta = std::atan2(ph, pw);
    roomba_image_location.radius = size_relative_diagonal * std::cos(theta);

    ROS_DEBUG_STREAM("x: " << roomba_image_location.x
                     << " y: " << roomba_image_location.y
                     << " r: " << roomba_image_location.radius
                     << " ray scale: " << ray_scale
                     << " size_relative_diagonal: " << size_relative_diagonal
                     << " theta: " << theta
                     << " cos(theta): " << std::cos(theta));
}

void RoombaEstimator::update(
        const cv::cuda::GpuMat& image,
        const ros::Time& time,
        std::vector<RoombaImageLocation>& roomba_image_locations)
{
    ROS_ASSERT(image.size() == input_size_);

    const auto start_time = std::chrono::high_resolution_clock::now();

    // Validation
    if(image.empty()) {
        iarc7_msgs::RoombaDetectionFrame result;
        result.header.stamp = time;
        result.header.frame_id = "map";
        result.camera_id = "bottom_camera";
        roomba_pub_.publish(result);
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    /// Fetch height
    //////////////////////////////////////////////////////////////////////////
    double height = getHeight(time);

    if (height < settings_.roomba_height + 0.01) {
        iarc7_msgs::RoombaDetectionFrame result;
        result.header.stamp = time;
        result.header.frame_id = "map";
        result.camera_id = "bottom_camera";
        roomba_pub_.publish(result);
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    /// Declare variables
    //////////////////////////////////////////////////////////////////////////
    std::vector<cv::RotatedRect> bounding_rects;
    std::vector<double> box_uncertainties;
    std::vector<double> flip_certainties;

    //////////////////////////////////////////////////////////////////////////
    /// Calculate the world field of view in meters and use to resize the image
    //////////////////////////////////////////////////////////////////////////
    geometry_msgs::Vector3Stamped a;
    geometry_msgs::Vector3Stamped b;
    pixelToRay(0, 0,          image.cols, image.rows, a);
    pixelToRay(0, image.cols, image.cols, image.rows, b);

    Eigen::Vector3d a_v (a.vector.x, a.vector.y, a.vector.z);
    Eigen::Vector3d b_v (b.vector.x, b.vector.y, b.vector.z);

    a_v /= a_v(2);
    b_v /= b_v(2);

    double distance = std::hypot(a_v(0) - b_v(0),
                                 a_v(1) - b_v(1));

    geometry_msgs::Vector3Stamped c;
    c.vector.z = 1;
    tf2::doTransform(c, c, camera_to_map_tf_);
    distance *= (height - settings_.roomba_height) / -c.vector.z;

    const auto boilerplate_time = std::chrono::high_resolution_clock::now();

    cv::cuda::GpuMat image_scaled;
    cv::cuda::resize(image, image_scaled, detection_size_);

    const auto resize_time = std::chrono::high_resolution_clock::now();

    //////////////////////////////////////////////////////////////////////////
    /// Run blob detection
    //////////////////////////////////////////////////////////////////////////
    blob_detector_->detect(image_scaled,
                           bounding_rects,
                           flip_certainties);

    const auto blob_time = std::chrono::high_resolution_clock::now();

    std::vector<std::array<double, 4>> position_covariances;
    calcBoxUncertainties(image_scaled.size(),
                         bounding_rects,
                         position_covariances,
                         box_uncertainties);

    ROS_ASSERT(box_uncertainties.size() == bounding_rects.size());
    ROS_ASSERT(flip_certainties.size() == bounding_rects.size());

    cv::Mat detected_rect_image;
    if (settings_.debug_detected_rects) {
        image_scaled.download(detected_rect_image);
    }

    //////////////////////////////////////////////////////////////////////////
    /// Fill out detection message
    /// And RoombaImageLocation vector
    //////////////////////////////////////////////////////////////////////////
    iarc7_msgs::RoombaDetectionFrame roomba_frame;
    roomba_frame.header.stamp = time;
    roomba_frame.header.frame_id = "map";
    roomba_frame.camera_id = "bottom_camera";

    for (unsigned int i = 0; i < bounding_rects.size(); i++) {
        if (box_uncertainties[i] < 0) {
            if (settings_.debug_detected_rects) {
                // Draw yellow rect for bad detection
                cv_utils::drawRotatedRect(detected_rect_image,
                                          bounding_rects[i],
                                          cv::Scalar(255, 255, 0));
            }
            continue;
        }

        cv::Point2f pos = bounding_rects[i].center;
        double angle = bounding_rects[i].angle * M_PI / 180;

        if (settings_.debug_detected_rects) {
            cv::Point2f p;
            p.x = pos.x + 100 * std::cos(angle);
            p.y = pos.y + 100 * std::sin(angle);
            cv::line(detected_rect_image, pos, p, cv::Scalar(255, 0, 0), 3);
            cv_utils::drawRotatedRect(detected_rect_image,
                                      bounding_rects[i],
                                      cv::Scalar(0, 0, 255));
        }

        iarc7_msgs::RoombaDetection roomba;
        RoombaImageLocation roomba_image_location;
        calcPose(pos,
                 angle,
                 image_scaled.cols,
                 image_scaled.rows,
                 roomba,
                 roomba_image_location);
        roomba.position_covariance[0] = position_covariances[i][0];
        roomba.position_covariance[1] = position_covariances[i][1];
        roomba.position_covariance[2] = position_covariances[i][2];
        roomba.position_covariance[3] = position_covariances[i][3];
        roomba.box_uncertainty = box_uncertainties[i];
        roomba.flip_certainty = flip_certainties[i];
        roomba_frame.roombas.push_back(roomba);
        roomba_image_locations.push_back(roomba_image_location);

        if (settings_.debug_detected_rects) {
            cv::Point2f p;
            p.x = roomba_image_location.x * image_scaled.cols;
            p.y = roomba_image_location.y * image_scaled.cols;
            //cv::circle(detected_rect_image,
            //           p,
            //           roomba_image_location.radius * image_scaled.cols,
            //           cv::Scalar(0, 255, 0));
        }
    }

    calcFloorPoly(roomba_frame.detection_region);

    if (settings_.debug_detected_rects) {
        const cv_bridge::CvImage cv_image {
            std_msgs::Header(),
            sensor_msgs::image_encodings::RGB8,
            detected_rect_image
        };

        debug_detected_rects_pub_.publish(cv_image.toImageMsg());
    }

    // publish
    roomba_pub_.publish(roomba_frame);

    const auto final_time = std::chrono::high_resolution_clock::now();

    const auto count = [](const auto& a, const auto& b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                b - a).count();
    };
    ROS_DEBUG_STREAM(
            "Boilerplate: " << count(start_time, boilerplate_time) << std::endl
         << "Resize: " << count(boilerplate_time, resize_time) << std::endl
         << "Detect: " << count(resize_time, blob_time) << std::endl
         << "Final: " << count(blob_time, final_time) << std::endl);
}

} // namespace iarc7_vision
