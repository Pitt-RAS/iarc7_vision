#include "iarc7_vision/RoombaEstimator.hpp"
#include <cmath>
#include <string>
#include <tf/transform_datatypes.h>

namespace iarc7_vision
{

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
}

void RoombaEstimator::CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg){
    bottom_camera.fromCameraInfo(msg);
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

void RoombaEstimator::CalcOdometry(cv::Point2f& pos, float angle, nav_msgs::Odometry& out, const ros::Time& time){
    geometry_msgs::Vector3 cam_pos = cam_tf.transform.translation;

    cv::Point3d cv_cam_ray = bottom_camera.projectPixelTo3dRay(pos);
    geometry_msgs::Vector3Stamped camera_ray;
    camera_ray.vector.x = cv_cam_ray.x;
    camera_ray.vector.y = cv_cam_ray.y;
    camera_ray.vector.z = cv_cam_ray.z;

    geometry_msgs::Vector3Stamped map_ray;
    tf2::doTransform(camera_ray, map_ray, cam_tf);

    float ray_scale = -(cam_pos.z - settings.roomba_height) / map_ray.vector.z;

    geometry_msgs::Point position;
    position.x = cam_pos.x + map_ray.vector.x * ray_scale;
    position.y = cam_pos.y + map_ray.vector.y * ray_scale;
    position.z = 0;

    out.header.frame_id = "map";
    out.header.stamp = time;
    out.pose.pose.position = position;
    out.pose.pose.orientation = tf::createQuaternionMsgFromYaw(angle * 3.141592653589 / 90.0);
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
void RoombaEstimator::PublishOdometry(){
    iarc7_msgs::OdometryArray msg;
    msg.data = odom_vector;
    roomba_pub.publish(msg);
}

void RoombaEstimator::update(const cv::Mat& image, const ros::Time& time){
    
    // Validation
    if(image.empty())
        return;

    if(!bottom_camera.initialized())
        return;

    float height = getHeight(time);

    if(height < 0.01)
        return;

    // Declare variables
    cv::vector<cv::Rect> boundRect;
    cv::Point2f pos = cv::Point2f();
    float angle = 0;

    // Calculate the world field of view in meters and use to resize the image
    cv::Point3d a = bottom_camera.projectPixelTo3dRay(cv::Point2d(0,0));
    cv::Point3d b = bottom_camera.projectPixelTo3dRay(cv::Point2d(0,image.cols));
    float distance = cv::sqrt((a.y-b.y)*(a.y-b.y) + (a.x-b.x)*(a.x-b.x));

    distance *= height;
    float desired_width = settings.pixels_per_meter * distance;
    float factor = desired_width / image.cols;

    cv::Mat frame = image.clone();
    cv::resize(frame, frame, cv::Size(), factor, factor);

    // Run blob detection
    bounder.detect(frame, boundRect);
    bounder.DilateBounds(frame, boundRect);

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
        pos *= 1 / factor;

        nav_msgs::Odometry odom;
        CalcOdometry(pos, angle, odom, time);
        ReportOdometry(odom);
        
    }


  // publish
  PublishOdometry();



  // DEBUGGING
  // for(unsigned int i=0;i<boundRect.size();i++){
  //     cv::rectangle(frame, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(255), 2, 8, 0);
  // }
  cv::imshow("Frame", frame);
  cv::waitKey(2);
}

}
