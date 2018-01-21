#include "iarc7_vision/RoombaEstimator.hpp"

#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <string>
#include <tf/transform_datatypes.h>

#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include "iarc7_vision/RoombaEstimatorConfig.h"

namespace iarc7_vision
{

/**
 * Converts a pixel in an image to a ray from the camera center
 *
 * @param[in]  px   X location of the pixel
 * @param[in]  py   Y location of the pixel
 * @param[in]  pw   Width of the image
 * @param[in]  ph   Height of the image
 * @param[out] ray  Vector pointing from the camera center to the pixel
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

    double camera_focal = -1;
    double camera_radius = camera_focal * pix_r / pix_focal;

    ray.vector.x = camera_radius * std::cos(theta);
    ray.vector.y = camera_radius * std::sin(theta);
    ray.vector.z = camera_focal;
}

RoombaEstimator::RoombaEstimator()
    : nh_(),
      private_nh_("~/roomba_estimator"),
      dynamic_reconfigure_server_(private_nh_),
      dynamic_reconfigure_settings_callback_(
              [this](iarc7_vision::RoombaEstimatorConfig& config, uint32_t) {
                  getDynamicSettings(config);
                  ght_detector_.onSettingsChanged();
              }),
      dynamic_reconfigure_called_(false),
      transform_wrapper_(),
      cam_tf_(),
      roomba_pub_(nh_.advertise<iarc7_msgs::OdometryArray>("roombas", 100)),
      odom_vector_(),
      settings_(getSettings(private_nh_)),
      blob_detector_(settings_, private_nh_),
      ght_detector_(settings_, private_nh_)
{
    dynamic_reconfigure_server_.setCallback(
            dynamic_reconfigure_settings_callback_);

    if (settings_.debug_ght_rects) {
        debug_ght_rects_pub_ = private_nh_.advertise<sensor_msgs::Image>("ght_rects", 10);
    }
}

void RoombaEstimator::getDynamicSettings(
        iarc7_vision::RoombaEstimatorConfig& config)
{
    if (!dynamic_reconfigure_called_) {
        config.ght_pos_thresh        = settings_.ght_pos_thresh;
        config.ght_angle_thresh      = settings_.ght_angle_thresh;
        config.ght_scale_thresh      = settings_.ght_scale_thresh;
        config.ght_canny_low_thresh  = settings_.ght_canny_low_thresh;
        config.ght_canny_high_thresh = settings_.ght_canny_high_thresh;
        config.ght_dp                = settings_.ght_dp;
        config.ght_levels            = settings_.ght_levels;
        config.ght_angle_step        = settings_.ght_angle_step;
        config.ght_scale_step        = settings_.ght_scale_step;

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
        settings_.ght_pos_thresh = config.ght_pos_thresh;
        settings_.ght_angle_thresh = config.ght_angle_thresh;
        settings_.ght_scale_thresh = config.ght_scale_thresh;
        settings_.ght_canny_low_thresh = config.ght_canny_low_thresh;
        settings_.ght_canny_high_thresh = config.ght_canny_high_thresh;
        settings_.ght_dp = config.ght_dp;
        settings_.ght_levels = config.ght_levels;
        settings_.ght_angle_step = config.ght_angle_step;
        settings_.ght_scale_step = config.ght_scale_step;

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
            "ght_pos_thresh",
            settings.ght_pos_thresh));
    ROS_ASSERT(private_nh.getParam(
            "ght_angle_thresh",
            settings.ght_angle_thresh));
    ROS_ASSERT(private_nh.getParam(
            "ght_scale_thresh",
            settings.ght_scale_thresh));
    ROS_ASSERT(private_nh.getParam(
            "ght_canny_low_thresh",
            settings.ght_canny_low_thresh));
    ROS_ASSERT(private_nh.getParam(
            "ght_canny_high_thresh",
            settings.ght_canny_high_thresh));
    ROS_ASSERT(private_nh.getParam(
            "ght_dp",
            settings.ght_dp));
    ROS_ASSERT(private_nh.getParam(
            "ght_levels",
            settings.ght_levels));
    ROS_ASSERT(private_nh.getParam(
            "ght_angle_step",
            settings.ght_angle_step));
    ROS_ASSERT(private_nh.getParam(
            "ght_scale_step",
            settings.ght_scale_step));
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
            "debug_rects",
            settings.debug_rects));
    ROS_ASSERT(private_nh.getParam(
            "debug_ght_rects",
            settings.debug_ght_rects));
    return settings;
}

void RoombaEstimator::odometryArrayCallback(
        const iarc7_msgs::OdometryArray& msg)
{
    odom_vector_ = msg.data;
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

void RoombaEstimator::calcOdometry(cv::Point2f& pos,
                                   double angle,
                                   double pw,
                                   double ph,
                                   nav_msgs::Odometry& out,
                                   const ros::Time& time)
{
    geometry_msgs::Vector3 cam_pos = cam_tf_.transform.translation;

    geometry_msgs::Vector3Stamped camera_ray;
    pixelToRay(pos.x, pos.y, pw, ph, camera_ray);

    geometry_msgs::Vector3Stamped map_ray;
    tf2::doTransform(camera_ray, map_ray, cam_tf_);

    double ray_scale = -(cam_pos.z - settings_.roomba_height)
                     / map_ray.vector.z;

    geometry_msgs::Point position;
    position.x = cam_pos.x + map_ray.vector.x * ray_scale;
    position.y = cam_pos.y + map_ray.vector.y * ray_scale;
    position.z = 0;

    geometry_msgs::Vector3 linear;
    linear.x = 0.3 * std::cos(angle);
    linear.y = 0.3 * std::sin(angle);
    linear.z = 0;

    out.header.frame_id = "map";
    out.header.stamp = time;
    out.pose.pose.position = position;
    out.pose.pose.orientation = tf::createQuaternionMsgFromYaw(angle);
    out.twist.twist.linear = linear;
}

// This will get totally screwed when it finds all 10 Roombas
void RoombaEstimator::reportOdometry(nav_msgs::Odometry& odom)
{
    int index = -1;
    double sq_tolerance = 0.1; // Roomba diameter is 0.254 meters
    if(odom_vector_.size()==10)
        sq_tolerance = 1000; // Impossible to attain, like my my love life
    for(unsigned int i=0;i<odom_vector_.size();i++){
        double xdiff = odom.pose.pose.position.x
                     - odom_vector_[i].pose.pose.position.x;
        double ydiff = odom.pose.pose.position.y
                     - odom_vector_[i].pose.pose.position.y;
        if(xdiff*xdiff + ydiff*ydiff < sq_tolerance){
            index = i;
            odom.child_frame_id = "roomba" + std::to_string(index);
            odom_vector_[i] = odom;
            continue;
        }
    }
    if(index==-1){
        index = odom_vector_.size();
        odom.child_frame_id = "roomba" + std::to_string(index);
        odom_vector_.push_back(odom);
    }
}

// Publish the most recent array of Roombas
void RoombaEstimator::publishOdometry()
{
    iarc7_msgs::OdometryArray msg;
    msg.data = odom_vector_;
    roomba_pub_.publish(msg);
}

void RoombaEstimator::update(const cv::cuda::GpuMat& image,
                             const ros::Time& time)
{
    // Validation
    if(image.empty())
        return;

    double height = getHeight(time);

    if(height < 0.01)
        return;

    // Declare variables
    std::vector<cv::Rect> bounding_rects;

    // Calculate the world field of view in meters and use to resize the image
    geometry_msgs::Vector3Stamped a;
    geometry_msgs::Vector3Stamped b;
    pixelToRay(0, 0,          image.cols, image.rows, a);
    pixelToRay(0, image.cols, image.cols, image.rows, b);

    double distance = std::hypot(a.vector.y - b.vector.y,
                                 a.vector.x - b.vector.x);

    distance *= height;
    double desired_width = settings_.template_pixels_per_meter * distance;
    double factor = desired_width / image.cols;

    cv::cuda::GpuMat image_scaled;
    cv::cuda::resize(image, image_scaled, cv::Size(), factor, factor);

    // Run blob detection
    blob_detector_.detect(image_scaled, bounding_rects);

    // Run the GHT on each blob
    //cv::Mat ght_rect_image;
    //if (settings_.debug_ght_rects) {
    //    image_scaled.download(ght_rect_image);
    //}

    //for(unsigned int i=0;i<bounding_rects.size();i++){
    //    cv::Point2f pos;
    //    double angle;

    //    bool detected = ght_detector_.detect(image_scaled,
    //                                         bounding_rects[i],
    //                                         pos,
    //                                         angle);

    //    if (!detected) continue;

    //    ROS_ERROR("Detected at position (%f %f), angle %f", pos.x, pos.y, angle);

    //    if (settings_.debug_ght_rects) {
    //        cv::Point2f p2;
    //        p2.x = pos.x + 100 * std::cos(angle);
    //        p2.y = pos.y + 100 * std::sin(angle);
    //        cv::line(ght_rect_image, pos, p2, cv::Scalar(255, 0, 0), 3);

    //        cv::RotatedRect rect;
    //        rect.center = pos;
    //        rect.size = cv::Size2f(50, 85);
    //        rect.angle = angle * 180.0 / M_PI;

    //        cv::Point2f pts[4];
    //        rect.points(pts);

    //        cv::line(ght_rect_image, pts[0], pts[1], cv::Scalar(0, 0, 255), 3);
    //        cv::line(ght_rect_image, pts[1], pts[2], cv::Scalar(0, 0, 255), 3);
    //        cv::line(ght_rect_image, pts[2], pts[3], cv::Scalar(0, 0, 255), 3);
    //        cv::line(ght_rect_image, pts[3], pts[0], cv::Scalar(0, 0, 255), 3);
    //    }

    //    nav_msgs::Odometry odom;
    //    calcOdometry(pos,
    //                 angle,
    //                 image_scaled.cols,
    //                 image_scaled.rows,
    //                 odom,
    //                 time);
    //    reportOdometry(odom);
    //}

    //if (settings_.debug_ght_rects) {
    //    const cv_bridge::CvImage cv_image {
    //        std_msgs::Header(),
    //        sensor_msgs::image_encodings::RGBA8,
    //        ght_rect_image
    //    };

    //    debug_ght_rects_pub_.publish(cv_image.toImageMsg());
    //}

    // publish
    publishOdometry();
}

} // namespace iarc7_vision
