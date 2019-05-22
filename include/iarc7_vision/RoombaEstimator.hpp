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
#include "iarc7_vision/RoombaImageLocation.hpp"

#include <sensor_msgs/CameraInfo.h>
#include <iarc7_msgs/RoombaDetection.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>

namespace iarc7_vision
{

/// Gets roomba positions from bottom camera images, converts to global
/// positions using tf, and publishes to /detected_roombas
class RoombaEstimator {
    public:
        RoombaEstimator(const cv::Size& image_size);

        /// Processes current frame and publishes detections
        ///
        /// @param[in]  image  Current frame to process (in rgb8)
        /// @param[in]  time   Timestamp of current frame
        /// @param[out]  roomba_image_locations Vector of roomba locations
        void update(const cv::cuda::GpuMat& image,
                    const ros::Time& time,
                    std::vector<RoombaImageLocation>&
                                roomba_image_locations);
    private:

        /// Converts a pixel in an image to a ray from the camera center
        ///
        /// @param[in]  px   X location of the pixel
        /// @param[in]  py   Y location of the pixel
        /// @param[in]  pw   Width of the image in pixels
        /// @param[in]  ph   Height of the image in pixels
        /// @param[out] ray  Unit vector pointing from the camera center to the pixel
        void pixelToRay(double px,
                        double py,
                        double pw,
                        double ph,
                        geometry_msgs::Vector3Stamped& ray) const;

        /// Callback for dynamic_reconfigure
        void getDynamicSettings(iarc7_vision::RoombaEstimatorConfig& config);

        /// Load settings from rosparam
        static RoombaEstimatorSettings getSettings(
                const ros::NodeHandle& private_nh);

        /// Fetch altitude of camera optical frame from tf
        ///
        /// Blocking
        ///
        /// @throws ros::Exception {Thrown if transform is not available at
        ///                         requested time}
        double getHeight(const ros::Time& time);

        void calcBoxUncertainties(
                const cv::Size& image_size,
                const std::vector<cv::RotatedRect>& bounding_rects,
                std::vector<std::array<double, 4>>& position_covariances,
                std::vector<double>& box_uncertainties) const;

        /// Calculate bounding polygon of the area of the floor that is
        /// currently visible to the camera
        void calcFloorPoly(geometry_msgs::Polygon& poly);

        /// Calculate roomba pose based on given location in the frame
        ///
        /// @param[in]  pos     Roomba position in pixels
        /// @param[in]  angle   Roomba angle
        /// @param[in]  pw      Image width in pixels
        /// @param[in]  ph      Image height in pixels
        /// @param[out] roomba  Roomba detection message
        //  @param[out] roomba_image_location Location of roomba in image with resolution indpendent scale
        void calcPose(const cv::Point2f& pos,
                      double angle,
                      double pw,
                      double ph,
                      iarc7_msgs::RoombaDetection& roomba,
                      RoombaImageLocation& roomba_image_location);

        ros::NodeHandle nh_;
        ros::NodeHandle private_nh_;

        dynamic_reconfigure::Server<iarc7_vision::RoombaEstimatorConfig>
            dynamic_reconfigure_server_;
        boost::function<void(iarc7_vision::RoombaEstimatorConfig&, uint32_t)>
            dynamic_reconfigure_settings_callback_;
        bool dynamic_reconfigure_called_;

        ros_utils::SafeTransformWrapper transform_wrapper_;
        geometry_msgs::TransformStamped camera_to_map_tf_;
        ros::Publisher roomba_pub_;

        RoombaEstimatorSettings settings_;

        const cv::Size input_size_;
        cv::Size detection_size_;
        std::unique_ptr<const RoombaBlobDetector> blob_detector_;

        ros::Publisher debug_detected_rects_pub_;
};

} // namespace iarc7_vision

#endif
