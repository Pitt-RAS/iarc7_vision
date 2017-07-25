#ifndef _IARC_VISION_ROOMBA_ESTIMATOR_HPP_
#define _IARC_VISION_ROOMBA_ESTIMATOR_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros_utils/SafeTransformWrapper.hpp>
#include <tf2_ros/transform_listener.h>

#include "iarc7_vision/RoombaBlob.hpp"
#include "iarc7_vision/RoombaGHT.hpp"

#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>
#include <iarc7_msgs/OdometryArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <mutex>


namespace iarc7_vision
{

// See vision_node_params.yaml for descriptions
struct RoombaEstimatorSettings {
    double pixels_per_meter;
    double roomba_plate_width;
    double roomba_height;
    int ght_levels;
    int ght_dp;
    int ght_votes_threshold;
    int camera_canny_threshold;
    int template_canny_threshold;
};

class RoombaEstimator{
    public:
        RoombaEstimator(ros::NodeHandle nh, const ros::NodeHandle& private_nh);
        void OdometryArrayCallback(const iarc7_msgs::OdometryArray& msg);
        void update(const cv::Mat& image, const ros::Time& time);
    private:
        void pixelToRay(double px, double py, double pw, double ph,
                        geometry_msgs::Vector3Stamped& ray);
        void getSettings(const ros::NodeHandle& private_nh);
        float getHeight(const ros::Time& time);
        void CalcOdometry(cv::Point2f& pos, double pw, double ph, float angle,
                          nav_msgs::Odometry& out, const ros::Time& time);
        void ReportOdometry(nav_msgs::Odometry& odom);
        void PublishOdometry();
        ros_utils::SafeTransformWrapper transform_wrapper_;
        geometry_msgs::TransformStamped cam_tf;
        ros::Publisher roomba_pub;
        std::vector<nav_msgs::Odometry> odom_vector;
        RoombaEstimatorSettings settings;
        RoombaBlob bounder;
        RoombaGHT ght;
        double bottom_camera_aov;
        std::mutex mtx;
};

}

#endif