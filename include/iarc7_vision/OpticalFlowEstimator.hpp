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

#include "ros_utils/LinearMsgInterpolator.hpp"
#include <ros_utils/SafeTransformWrapper.hpp>
#include <tf2_ros/transform_listener.h>

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>

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
    double scale_factor;
    double imu_update_timeout;
    double variance;
    double variance_scale;
    double x_cutoff_region_velocity_measurement;
    double y_cutoff_region_velocity_measurement;
    int debug_frameskip;
    double orientation_image_time_offset;
};

struct OpticalFlowDebugSettings {
    bool debug_vectors_image;
    bool debug_average_vector_image;
};

class OpticalFlowEstimator {
  public:
    OpticalFlowEstimator(ros::NodeHandle nh,
                         const OpticalFlowEstimatorSettings& flow_estimator_settings,
                         const OpticalFlowDebugSettings& debug_settings);
    void update(const sensor_msgs::Image::ConstPtr& message);

    bool waitUntilReady(const ros::Duration& startup_timeout);

  private:

    /// Compute the focal length (in px) from image size and dfov
    ///
    /// @param[in] fov Field of view in radians
    static double getFocalLength(const cv::Size& img_size, double fov);

    void estimateVelocity(geometry_msgs::TwistWithCovarianceStamped& velocity,
                                         const cv::Mat& image,
                                         double height,
                                         ros::Time time);

    void updateFilteredPosition(const ros::Time& time);

    cv::Point2f findAverageVector(const std::vector<cv::Point2f>& prevPts,
                                  const std::vector<cv::Point2f>& nextPts,
                                  const std::vector<uchar>& status,
                                  const double x_cutoff,
                                  const double y_cutoff,
                                  const cv::Size& image_size);

    const OpticalFlowEstimatorSettings& flow_estimator_settings_;

    const OpticalFlowDebugSettings& debug_settings_;

    geometry_msgs::PointStamped last_filtered_position_;

    ros::Publisher twist_pub_;

    ros_utils::SafeTransformWrapper transform_wrapper_;

    cv::gpu::GpuMat last_scaled_image_;

    cv::gpu::GpuMat last_scaled_grayscale_image_;

    ros::Publisher debug_velocity_vector_image_pub_;

    ros::Publisher debug_average_velocity_vector_image_pub_;

    ros_utils::LinearMsgInterpolator<
       sensor_msgs::Imu,
       tf2::Vector3>
          imu_interpolator_;

    tf2::Vector3 last_angular_velocity_;

    geometry_msgs::TransformStamped last_filtered_transform_stamped_;

    ros::Time last_message_time_;

    ros::Publisher ori_pub;
    ros::Publisher correction_pub_;
    ros::Publisher raw_pub_;

    double last_p_ = 0;
    double last_r_ = 0;
};

} // namespace iarc7_vision

#endif // include guard
