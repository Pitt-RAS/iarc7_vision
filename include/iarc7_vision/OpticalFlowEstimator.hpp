#ifndef IARC7_VISION_OPTICAL_FLOW_ESTIMATOR_HPP_
#define IARC7_VISION_OPTICAL_FLOW_ESTIMATOR_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>

#include <ros_utils/SafeTransformWrapper.hpp>
#include <tf2_ros/transform_listener.h>

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

namespace iarc7_vision {

struct OpticalFlowEstimatorSettings {
    double pixels_per_meter;
    double fov;
    double min_estimation_altitude;
    int win_size;
    int max_level;
    int iters;
    int points;
    double quality_level;
    int min_dist;
};

struct OpticalFlowDebugSettings {

};

class OpticalFlowEstimator {
  public:
    OpticalFlowEstimator(const OpticalFlowEstimatorSettings& flow_estimator_settings,
                         const OpticalFlowDebugSettings& debug_settings);
    void update(const sensor_msgs::Image::ConstPtr& message);

  private:

    /// Compute the focal length (in px) from image size and dfov
    ///
    /// @param[in] fov Field of view in radians
    static double getFocalLength(const cv::Size& img_size, double fov);

    void estimateVelocity(geometry_msgs::TwistWithCovarianceStamped& velocity,
                                         const cv::Mat& last_image,
                                         const cv::Mat& image,
                                         double height) const;

    void updateFilteredPosition(const ros::Time& time);

    const OpticalFlowEstimatorSettings& flow_estimator_settings_;

    const OpticalFlowDebugSettings& debug_settings_;

    geometry_msgs::PointStamped last_filtered_position_;

    ros::Publisher twist_pub_;

    ros_utils::SafeTransformWrapper transform_wrapper_;

    sensor_msgs::Image::ConstPtr last_message_;
};

} // namespace iarc7_vision

#endif // include guard
