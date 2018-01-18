#ifndef _IARC_VISION_ROOMBA_ESTIMATOR_HPP_
#define _IARC_VISION_ROOMBA_ESTIMATOR_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <dynamic_reconfigure/server.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros_utils/SafeTransformWrapper.hpp>
#include <tf2_ros/transform_listener.h>

#include "iarc7_vision/RoombaBlobDetector.hpp"
#include "iarc7_vision/RoombaEstimatorConfig.h"
#include "iarc7_vision/RoombaEstimatorSettings.hpp"
#include "iarc7_vision/RoombaGHT.hpp"

#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>
#include <iarc7_msgs/OdometryArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>

namespace iarc7_vision
{

class RoombaEstimator {
    public:
        RoombaEstimator(ros::NodeHandle nh, ros::NodeHandle& private_nh);

        void odometryArrayCallback(const iarc7_msgs::OdometryArray& msg);
        void update(const cv::cuda::GpuMat& image, const ros::Time& time);
    private:
        void pixelToRay(double px, double py, double pw, double ph,
                        geometry_msgs::Vector3Stamped& ray);
        void getDynamicSettings(iarc7_vision::RoombaEstimatorConfig& config);
        static RoombaEstimatorSettings getSettings(
                const ros::NodeHandle& private_nh);
        float getHeight(const ros::Time& time);
        void calcOdometry(cv::Point2f& pos, double pw, double ph, float angle,
                          nav_msgs::Odometry& out, const ros::Time& time);
        void reportOdometry(nav_msgs::Odometry& odom);
        void publishOdometry();

        dynamic_reconfigure::Server<iarc7_vision::RoombaEstimatorConfig>
            dynamic_reconfigure_server_;
        boost::function<void(iarc7_vision::RoombaEstimatorConfig&, uint32_t)>
            dynamic_reconfigure_settings_callback_;
        bool dynamic_reconfigure_called_;

        ros_utils::SafeTransformWrapper transform_wrapper_;
        geometry_msgs::TransformStamped cam_tf_;
        ros::Publisher roomba_pub_;
        std::vector<nav_msgs::Odometry> odom_vector_;

        RoombaEstimatorSettings settings_;
        RoombaBlobDetector blob_detector_;
        RoombaGHT ght_detector_;

        ros::Publisher debug_ght_rects_pub_;
};

} // namespace iarc7_vision

#endif
