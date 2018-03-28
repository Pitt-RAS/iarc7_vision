#include "iarc7_vision/RoombaEstimator.hpp"

#include <chrono>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <string>
#include <tf/transform_datatypes.h>

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

/**
 * Converts a pixel in an image to a ray from the camera center
 *
 * @param[in]  px   X location of the pixel
 * @param[in]  py   Y location of the pixel
 * @param[in]  pw   Width of the image
 * @param[in]  ph   Height of the image
 * @param[out] ray  Unit vector pointing from the camera center to the pixel
 */
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

        settings_.morphology_size = config.morphology_size;
        settings_.morphology_iterations = config.morphology_iterations;
    }
}

RoombaEstimatorSettings RoombaEstimator::getSettings(
        const ros::NodeHandle& private_nh)
{
    RoombaEstimatorSettings settings;
    ROS_ASSERT(private_nh.getParam(
            "template_pixels_per_meter",
            settings.template_pixels_per_meter));
    ROS_ASSERT(private_nh.getParam(
            "roomba_plate_width",
            settings.roomba_plate_width));
    ROS_ASSERT(private_nh.getParam(
            "roomba_height",
            settings.roomba_height));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_h_green_min",
            settings.hsv_slice_h_green_min));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_h_green_max",
            settings.hsv_slice_h_green_max));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_h_red1_min",
            settings.hsv_slice_h_red1_min));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_h_red1_max",
            settings.hsv_slice_h_red1_max));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_h_red2_min",
            settings.hsv_slice_h_red2_min));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_h_red2_max",
            settings.hsv_slice_h_red2_max));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_s_min",
            settings.hsv_slice_s_min));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_s_max",
            settings.hsv_slice_s_max));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_v_min",
            settings.hsv_slice_v_min));
    ROS_ASSERT(private_nh.getParam(
            "hsv_slice_v_max",
            settings.hsv_slice_v_max));
    ROS_ASSERT(private_nh.getParam(
            "morphology_size",
            settings.morphology_size));
    ROS_ASSERT(private_nh.getParam(
            "morphology_iterations",
            settings.morphology_iterations));
    ROS_ASSERT(private_nh.getParam(
            "morphology_size",
            settings.morphology_size));
    ROS_ASSERT(private_nh.getParam(
            "bottom_camera_aov",
            settings.bottom_camera_aov));
    ROS_ASSERT(private_nh.getParam(
            "debug_hsv_slice",
            settings.debug_hsv_slice));
    ROS_ASSERT(private_nh.getParam(
            "debug_contours",
            settings.debug_contours));
    ROS_ASSERT(private_nh.getParam(
            "debug_detected_rects",
            settings.debug_detected_rects));
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
                               iarc7_msgs::RoombaDetection& roomba)
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
    roomba.pose.theta = angle;
}

void RoombaEstimator::update(const cv::cuda::GpuMat& image,
                             const ros::Time& time)
{
    // Validation
    if(image.empty())
        return;

    double height = getHeight(time);

    if (height < settings_.roomba_height + 0.01) {
        return;
    }

    // Declare variables
    std::vector<cv::RotatedRect> bounding_rects;

    // Calculate the world field of view in meters and use to resize the image
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

    // Run blob detection
    blob_detector_.detect(image_scaled, bounding_rects);

    cv::Mat detected_rect_image;
    if (settings_.debug_detected_rects) {
        image_scaled.download(detected_rect_image);
    }

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
        calcPose(pos,
                 angle,
                 image_scaled.cols,
                 image_scaled.rows,
                 roomba);
        roomba_frame.roombas.push_back(roomba);
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
