#include "iarc7_vision/RoombaEstimator.hpp"
#include <cmath>
#include <string>
#include <tf/transform_datatypes.h>

namespace iarc7_vision
{

void RoombaEstimator::pixelToRay(double px, double py, double pw, double ph, geometry_msgs::Vector3Stamped& ray){
    px -= pw * 0.5;
    py -= ph * 0.5;

    double pix_r = std::sqrt( px*px + py*py );
    double pix_R = std::sqrt( ph*ph + pw*pw ) * 0.5;

    double max_phi = bottom_camera_aov * 3.141592653589 / 360;
    double pix_focal = pix_R / tan( max_phi );
    double theta = atan2( py, px );

    double camera_focal = -1;
    double camera_radius = camera_focal * pix_r / pix_focal;

    ray.vector.x = camera_radius * cos( theta );
    ray.vector.y = camera_radius * sin( theta );
    ray.vector.z = camera_focal;
}

RoombaEstimator::RoombaEstimator(ros::NodeHandle nh,
                  const ros::NodeHandle& private_nh) : transform_wrapper_()
{
    roomba_pub = nh.advertise<iarc7_msgs::OdometryArray>("roombas", 100);
    getSettings(private_nh);
    ght.setup(settings.pixels_per_meter, settings.roomba_plate_width,
              settings.ght_levels, settings.ght_dp,
              settings.ght_votes_threshold, settings.template_canny_threshold);
}

void RoombaEstimator::getSettings(const ros::NodeHandle& private_nh){
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
            bottom_camera_aov));
}

void RoombaEstimator::OdometryArrayCallback(const iarc7_msgs::OdometryArray& msg){
    mtx.lock();
    odom_vector = msg.data;
    mtx.unlock();
    ROS_ERROR_STREAM( "Odom Array Updated.");
}

float RoombaEstimator::getHeight(const ros::Time& time){
    if (!transform_wrapper_.getTransformAtTime(cam_tf,
                                               "map",
                                               "bottom_camera_optical",
                                               time,
                                               ros::Duration(1.0))) {
        throw ros::Exception("Failed to fetch transform");
    }
    return cam_tf.transform.translation.z;
}

void RoombaEstimator::CalcOdometry(cv::Point2f& pos, double pw, double ph, float angle, nav_msgs::Odometry& out, const ros::Time& time){
    float rads = angle * 3.141592653589 / 90.0;
    geometry_msgs::Vector3 cam_pos = cam_tf.transform.translation;

    geometry_msgs::Vector3Stamped camera_ray;
    pixelToRay(pos.x, pos.y, pw, ph, camera_ray);

    geometry_msgs::Vector3Stamped map_ray;
    tf2::doTransform(camera_ray, map_ray, cam_tf);

    float ray_scale = -(cam_pos.z - settings.roomba_height) / map_ray.vector.z;

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
void RoombaEstimator::ReportOdometry(nav_msgs::Odometry& odom){
    int index = -1;
    float sq_tolerance = 0.1; // Roomba diameter is 0.254 meters
    if(odom_vector.size()==10)
        sq_tolerance = 1000; // Impossible to attain, like my my love life
    for(unsigned int i=0;i<odom_vector.size();i++){
        float xdiff = odom.pose.pose.position.x - odom_vector[i].pose.pose.position.x;
        float ydiff = odom.pose.pose.position.y - odom_vector[i].pose.pose.position.y;
        if(xdiff*xdiff + ydiff*ydiff < sq_tolerance){
            index = i;
            odom.child_frame_id = "roomba" + std::to_string(index);
            odom_vector[i] = odom;
            continue;
        }
    }
    if(index==-1){
        index = odom_vector.size();
        odom.child_frame_id = "roomba" + std::to_string(index);
        odom_vector.push_back(odom);
    }
}

// Publish the most recent array of Roombas
void RoombaEstimator::PublishOdometry()
{
    iarc7_msgs::OdometryArray msg;
    msg.data = odom_vector;
    roomba_pub.publish(msg);
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
    cv::vector<cv::Rect> boundRect;
    cv::Point2f pos = cv::Point2f();
    float angle = 0;

    // Calculate the world field of view in meters and use to resize the image
    geometry_msgs::Vector3Stamped a;
    geometry_msgs::Vector3Stamped b;
    pixelToRay(0,0,image.cols,image.rows,a);
    pixelToRay(0,image.cols,image.cols,image.rows,b);

    float distance = std::sqrt((a.vector.y-b.vector.y)*(a.vector.y-b.vector.y)
                            + (a.vector.x-b.vector.x)*(a.vector.x-b.vector.x));

    distance *= height;
    float desired_width = settings.pixels_per_meter * distance;
    float factor = desired_width / image.cols;

    cv::Mat frame = image.clone();
    cv::resize(frame, frame, cv::Size(), factor, factor);

    // Run blob detection
    bounder.detect(frame, boundRect);
    bounder.DilateBounds(frame, boundRect);

    mtx.lock();
    // Run the GHT on each blob
    for(unsigned int i=0;i<boundRect.size();i++){
        angle = ght.detect(frame, boundRect[i], pos,
                           settings.camera_canny_threshold);

        cv::Point2f P2;
        P2.x =  (int)round(pos.x + 100 * cos(angle * CV_PI / 180.0));
        P2.y =  (int)round(pos.y + 100 * sin(angle * CV_PI / 180.0));
        line(frame, pos, P2, cv::Scalar(255, 0, 0), 3);
        
        cv::RotatedRect rect;
        rect.center = pos;
        rect.size = cv::Size2f(50, 85);
        rect.angle = angle;

        cv::Point2f pts[4];
        rect.points(pts);

        cv::line(frame, pts[0], pts[1], cv::Scalar(0, 0, 255), 3);
        cv::line(frame, pts[1], pts[2], cv::Scalar(0, 0, 255), 3);
        cv::line(frame, pts[2], pts[3], cv::Scalar(0, 0, 255), 3);
        cv::line(frame, pts[3], pts[0], cv::Scalar(0, 0, 255), 3);

        if(angle == -1) continue;
        // divide by factor to convert coordinates back to original scaling

        nav_msgs::Odometry odom;
        CalcOdometry(pos, frame.cols, frame.rows, angle, odom, time);
        ReportOdometry(odom);
        
    }

  // publish
  PublishOdometry();
  mtx.unlock();

  cv::imshow("Frame", frame);
  cv::waitKey(2);
}

}
