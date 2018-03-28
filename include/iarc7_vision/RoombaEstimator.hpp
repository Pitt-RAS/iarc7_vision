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

#include <sensor_msgs/CameraInfo.h>
#include <iarc7_msgs/RoombaDetection.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>

namespace iarc7_vision
{

class RoombaEstimator {
    public:
        RoombaEstimator();

        void update(const cv::cuda::GpuMat& image, const ros::Time& time);
    private:
        void pixelToRay(double px,
                        double py,
                        double pw,
                        double ph,
                        geometry_msgs::Vector3Stamped& ray);

        void getDynamicSettings(iarc7_vision::RoombaEstimatorConfig& config);

        static RoombaEstimatorSettings getSettings(
                const ros::NodeHandle& private_nh);

        double getHeight(const ros::Time& time);

        void calcFloorPoly(geometry_msgs::Polygon& poly);

        void calcPose(const cv::Point2f& pos,
                      double angle,
                      double pw,
                      double ph,
                      iarc7_msgs::RoombaDetection& roomba);

        ros::NodeHandle nh_;
        ros::NodeHandle private_nh_;

        dynamic_reconfigure::Server<iarc7_vision::RoombaEstimatorConfig>
            dynamic_reconfigure_server_;
        boost::function<void(iarc7_vision::RoombaEstimatorConfig&, uint32_t)>
            dynamic_reconfigure_settings_callback_;
        bool dynamic_reconfigure_called_;

        ros_utils::SafeTransformWrapper transform_wrapper_;
        geometry_msgs::TransformStamped cam_tf_;
        ros::Publisher roomba_pub_;

        RoombaEstimatorSettings settings_;
        RoombaBlobDetector blob_detector_;

        ros::Publisher debug_detected_rects_pub_;
};

} // namespace iarc7_vision

#endif
