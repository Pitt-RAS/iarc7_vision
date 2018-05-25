#include "iarc7_vision/RoombaEstimator.hpp"

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

RoombaEstimator::RoombaEstimator()
    : nh_(),
      private_nh_("~/roomba_estimator"),
      dynamic_reconfigure_server_(private_nh_),
      dynamic_reconfigure_settings_callback_(
              [this](iarc7_vision::RoombaEstimatorConfig& config, uint32_t) {
                  getDynamicSettings(config);
              }),
      dynamic_reconfigure_called_(false),
      transform_wrapper_(),
      cam_tf_(),
      roomba_pub_(nh_.advertise<iarc7_msgs::RoombaDetectionFrame>(
                  "detected_roombas", 100)),
      settings_(getSettings(private_nh_)),
      blob_detector_(settings_, private_nh_)
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
                                 geometry_msgs::Vector3Stamped& ray)
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
    IARC7_VISION_RES_LOAD(template_pixels_per_meter);
    IARC7_VISION_RES_LOAD(roomba_plate_width);
    IARC7_VISION_RES_LOAD(roomba_height);
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
                cam_tf_,
                "map",
                "bottom_camera_rgb_optical_frame",
                time,
                ros::Duration(1.0))) {
        throw ros::Exception("Failed to fetch transform");
    }
    return cam_tf_.transform.translation.z;
}

void RoombaEstimator::calcFloorPoly(geometry_msgs::Polygon& poly)
{
    poly.points.clear();

    geometry_msgs::Vector3Stamped camera_ray;
    geometry_msgs::Point32 point;
    double scale;

    geometry_msgs::PointStamped camera_position;
    tf2::doTransform(camera_position, camera_position, cam_tf_);

    for (const auto& image_point : {
            std::make_pair(0, 0),
            std::make_pair(0, 1),
            std::make_pair(1, 1),
            std::make_pair(1, 0)}) {
        pixelToRay(image_point.first, image_point.second, 1, 1, camera_ray);
        tf2::doTransform(camera_ray, camera_ray, cam_tf_);
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
    tf2::doTransform(cam_pos, cam_pos, cam_tf_);

    geometry_msgs::Vector3Stamped camera_ray;
    pixelToRay(pos.x, pos.y, pw, ph, camera_ray);

    geometry_msgs::Vector3Stamped map_ray;
    tf2::doTransform(camera_ray, map_ray, cam_tf_);

    double ray_scale = (cam_pos.point.z - settings_.roomba_height)
                     / -map_ray.vector.z;

    roomba.pose.x = cam_pos.point.x + map_ray.vector.x * ray_scale;
    roomba.pose.y = cam_pos.point.y + map_ray.vector.y * ray_scale;

    geometry_msgs::Vector3Stamped roomba_yaw;
    geometry_msgs::Vector3Stamped roomba_yaw_after;
    roomba_yaw.vector.x = std::cos(angle);
    roomba_yaw.vector.y = std::sin(angle);
    tf2::doTransform(roomba_yaw, roomba_yaw_after, cam_tf_);
    //ROS_ERROR_STREAM(roomba_yaw);
    //ROS_ERROR_STREAM(roomba_yaw_after);

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

void RoombaEstimator::update(const cv::cuda::GpuMat& image,
                             const ros::Time& time,
                             std::vector<RoombaImageLocation>&
                                roomba_image_locations)
{
    // Validation
    if(image.empty())
        return;

    //////////////////////////////////////////////////////////////////////////
    /// Fetch height
    //////////////////////////////////////////////////////////////////////////
    double height = getHeight(time);

    if (height < settings_.roomba_height + 0.01) {
        return;
    }

    //////////////////////////////////////////////////////////////////////////
    /// Declare variables
    //////////////////////////////////////////////////////////////////////////
    std::vector<cv::RotatedRect> bounding_rects;

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
    tf2::doTransform(c, c, cam_tf_);
    distance *= (height - settings_.roomba_height) / -c.vector.z;

    double desired_width = settings_.template_pixels_per_meter * distance;
    double factor = desired_width / image.cols;

    cv::cuda::GpuMat image_scaled;
    cv::cuda::resize(image, image_scaled, cv::Size(), factor, factor);

    //////////////////////////////////////////////////////////////////////////
    /// Run blob detection
    //////////////////////////////////////////////////////////////////////////
    blob_detector_.detect(image_scaled, bounding_rects);

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

    for (unsigned int i = 0; i < bounding_rects.size(); i++) {
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
        roomba_frame.roombas.push_back(roomba);
        roomba_image_locations.push_back(roomba_image_location);

        if (settings_.debug_detected_rects) {
            cv::Point2f p;
            p.x = roomba_image_location.x * image_scaled.cols;
            p.y = roomba_image_location.y * image_scaled.cols;
            cv::circle(detected_rect_image,
                       p,
                       roomba_image_location.radius * image_scaled.cols,
                       cv::Scalar(0, 255, 0));
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
}

} // namespace iarc7_vision
