#include "iarc7_vision/RoombaEstimator.hpp"
#include <cmath>
#include <string>
#include <tf/transform_datatypes.h>

namespace iarc7_vision
{

void RoombaEstimator::pixelToRay(double px,
                                 double py,
                                 double pw,
                                 double ph,
                                 geometry_msgs::Vector3Stamped& ray)
{
    px -= pw * 0.5;
    py -= ph * 0.5;

    double pix_r = std::sqrt( px*px + py*py );
    double pix_R = std::sqrt( ph*ph + pw*pw ) * 0.5;

    double max_phi = settings_.bottom_camera_aov * 3.141592653589 / 360;
    double pix_focal = pix_R / tan( max_phi );
    double theta = atan2( py, px );

    double camera_focal = -1;
    double camera_radius = camera_focal * pix_r / pix_focal;

    ray.vector.x = camera_radius * cos( theta );
    ray.vector.y = camera_radius * sin( theta );
    ray.vector.z = camera_focal;
}

RoombaEstimator::RoombaEstimator(ros::NodeHandle nh,
                                 ros::NodeHandle& private_nh)
    : transform_wrapper_(),
      cam_tf_(),
      roomba_pub_(nh.advertise<iarc7_msgs::OdometryArray>("roombas", 100)),
      odom_vector_(),
      settings_(getSettings(private_nh)),
      blob_detector_(settings_, private_nh),
      mtx_()
{
    //ght.setup(settings.pixels_per_meter, settings.roomba_plate_width,
    //          settings.ght_levels, settings.ght_dp,
    //          settings.ght_votes_threshold, settings.template_canny_threshold);
}

RoombaEstimatorSettings RoombaEstimator::getSettings(
        const ros::NodeHandle& private_nh)
{
    RoombaEstimatorSettings settings;
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/pixels_per_meter",
            settings.pixels_per_meter));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/roomba_plate_width",
            settings.roomba_plate_width));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/ght_levels",
            settings.ght_levels));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/ght_dp",
            settings.ght_dp));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/ght_votes_threshold",
            settings.ght_votes_threshold));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/camera_canny_threshold",
            settings.camera_canny_threshold));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/template_canny_threshold",
            settings.template_canny_threshold));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/bottom_camera_aov",
            settings.bottom_camera_aov));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/debug_hsv_slice",
            settings.debug_hsv_slice));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/debug_contours",
            settings.debug_contours));
    ROS_ASSERT(private_nh.getParam(
            "roomba_estimator_settings/debug_rects",
            settings.debug_rects));
    return settings;
}

void RoombaEstimator::odometryArrayCallback(
        const iarc7_msgs::OdometryArray& msg)
{
    mtx_.lock();
    odom_vector_ = msg.data;
    mtx_.unlock();
}

float RoombaEstimator::getHeight(const ros::Time& time)
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
                                   double pw,
                                   double ph,
                                   float angle,
                                   nav_msgs::Odometry& out,
                                   const ros::Time& time)
{
    float rads = angle * M_PI / 90.0;
    geometry_msgs::Vector3 cam_pos = cam_tf_.transform.translation;

    geometry_msgs::Vector3Stamped camera_ray;
    pixelToRay(pos.x, pos.y, pw, ph, camera_ray);

    geometry_msgs::Vector3Stamped map_ray;
    tf2::doTransform(camera_ray, map_ray, cam_tf_);

    float ray_scale = -(cam_pos.z - settings_.roomba_height) / map_ray.vector.z;

    geometry_msgs::Point position;
    position.x = cam_pos.x + map_ray.vector.x * ray_scale;
    position.y = cam_pos.y + map_ray.vector.y * ray_scale;
    position.z = 0;

    geometry_msgs::Vector3 linear;
    linear.x = 0.3 * cos(rads);
    linear.y = 0.3 * sin(rads);
    linear.z = 0;

    out.header.frame_id = "map";
    out.header.stamp = time;
    out.pose.pose.position = position;
    out.pose.pose.orientation = tf::createQuaternionMsgFromYaw(rads);
    out.twist.twist.linear = linear;
}

// This will get totally screwed when it finds all 10 Roombas
void RoombaEstimator::reportOdometry(nav_msgs::Odometry& odom)
{
    int index = -1;
    float sq_tolerance = 0.1; // Roomba diameter is 0.254 meters
    if(odom_vector_.size()==10)
        sq_tolerance = 1000; // Impossible to attain, like my my love life
    for(unsigned int i=0;i<odom_vector_.size();i++){
        float xdiff = odom.pose.pose.position.x - odom_vector_[i].pose.pose.position.x;
        float ydiff = odom.pose.pose.position.y - odom_vector_[i].pose.pose.position.y;
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

    float height = getHeight(time);

    if(height < 0.01)
        return;

    // Declare variables
    std::vector<cv::Rect> boundRect;
    //cv::Point2f pos = cv::Point2f();
    //float angle = 0;

    // Calculate the world field of view in meters and use to resize the image
    geometry_msgs::Vector3Stamped a;
    geometry_msgs::Vector3Stamped b;
    pixelToRay(0,0,image.cols,image.rows,a);
    pixelToRay(0,image.cols,image.cols,image.rows,b);

    float distance = std::sqrt((a.vector.y-b.vector.y)*(a.vector.y-b.vector.y)
                            + (a.vector.x-b.vector.x)*(a.vector.x-b.vector.x));

    distance *= height;
    float desired_width = settings_.pixels_per_meter * distance;
    float factor = desired_width / image.cols;

    cv::cuda::GpuMat image_scaled;
    cv::cuda::resize(image, image_scaled, cv::Size(), factor, factor);

    // Run blob detection
    blob_detector_.detect(image_scaled, boundRect);

    mtx_.lock();
    // Run the GHT on each blob
    //for(unsigned int i=0;i<boundRect.size();i++){
    //    angle = ght.detect(image, boundRect[i], pos,
    //                       settings.camera_canny_threshold);

    //    cv::Point2f P2;
    //    P2.x =  (int)round(pos.x + 100 * cos(angle * CV_PI / 180.0));
    //    P2.y =  (int)round(pos.y + 100 * sin(angle * CV_PI / 180.0));
    //    line(image, pos, P2, cv::Scalar(255, 0, 0), 3);

    //    cv::RotatedRect rect;
    //    rect.center = pos;
    //    rect.size = cv::Size2f(50, 85);
    //    rect.angle = angle;

    //    cv::Point2f pts[4];
    //    rect.points(pts);

    //    cv::line(image, pts[0], pts[1], cv::Scalar(0, 0, 255), 3);
    //    cv::line(image, pts[1], pts[2], cv::Scalar(0, 0, 255), 3);
    //    cv::line(image, pts[2], pts[3], cv::Scalar(0, 0, 255), 3);
    //    cv::line(image, pts[3], pts[0], cv::Scalar(0, 0, 255), 3);

    //    if(angle == -1) continue;
    //    // divide by factor to convert coordinates back to original scaling

    //    nav_msgs::Odometry odom;
    //    CalcOdometry(pos, image.cols, image.rows, angle, odom, time);
    //    ReportOdometry(odom);

    //}

  // publish
  publishOdometry();
  mtx_.unlock();
}

}
