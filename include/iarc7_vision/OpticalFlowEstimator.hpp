#ifndef IARC7_VISION_OPTICAL_FLOW_ESTIMATOR_HPP_
#define IARC7_VISION_OPTICAL_FLOW_ESTIMATOR_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <ros_utils/SafeTransformWrapper.hpp>

#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

namespace iarc7_vision {

struct OpticalFlowEstimatorSettings {
    double fov;
    double min_estimation_altitude;
    int win_size;
    int max_level;
    int iters;
    int points;
    double quality_level;
    int min_dist;
    double scale_factor;
    double variance;
    double variance_scale;
    double x_cutoff_region_velocity_measurement;
    double y_cutoff_region_velocity_measurement;
    int debug_frameskip;
    double tf_timeout;
};

struct OpticalFlowDebugSettings {
    bool debug_vectors_image;
    bool debug_average_vector_image;
    bool debug_times;
};

class OpticalFlowEstimator {
  public:

    //////////////////
    // CONSTRUCTORS //
    //////////////////

    OpticalFlowEstimator(
            const OpticalFlowEstimatorSettings& flow_estimator_settings,
            const OpticalFlowDebugSettings& debug_settings);

    ////////////////////
    // PUBLIC METHODS //
    ////////////////////

    bool __attribute__((warn_unused_result)) onSettingsChanged();

    void update(const sensor_msgs::Image::ConstPtr& message);

    bool __attribute__((warn_unused_result)) waitUntilReady(
            const ros::Duration& startup_timeout);

  private:

    /////////////////////
    // PRIVATE METHODS //
    /////////////////////

    /// Calculate new velocity estimate
    ///
    /// @param[in]  average_vec {Movement since last frame, averaged over all
    ///                          features}
    /// @param[in]  height      Altitude of the camera at `time`
    /// @param[in]  time        Timestamp of the image
    ///
    /// @return                 Velocity estimate
    geometry_msgs::TwistWithCovarianceStamped estimateVelocity(
            const cv::Point2f& average_vec,
            const tf2::Quaternion& curr_orientation,
            const tf2::Quaternion& last_orientation,
            double height,
            const ros::Time& time) const;

    /// Finds the average of the given vectors, filtering out points outside
    /// of the cutoff region
    ///
    /// @param[in] tails      Tails of the input vectors
    /// @param[in] heads      Heads of the input vectors
    /// @param[in] status     Status of each vector, 1 if valid and 0 otherwise
    /// @param[in] x_cutoff   {Width of region to be discarded on each side as a
    ///                        percentage of the total width}
    /// @param[in] y_cutoff   {Height of region to be discarded on each side as
    ///                        a percentage of the total height}
    /// @param[in] image_size {Size of image that the points are from, used for
    ///                        calculating cutoffs}
    ///
    /// @return               Average of valid input vectors
    static cv::Point2f findAverageVector(const std::vector<cv::Point2f>& tails,
                                         const std::vector<cv::Point2f>& heads,
                                         const std::vector<uchar>& status,
                                         const double x_cutoff,
                                         const double y_cutoff,
                                         const cv::Size& image_size);

    void findFeatureVectors(const cv::gpu::GpuMat& curr_frame,
                            const cv::gpu::GpuMat& curr_gray_frame,
                            const cv::gpu::GpuMat& last_frame,
                            const cv::gpu::GpuMat& last_gray_frame,
                            std::vector<cv::Point2f>& tails,
                            std::vector<cv::Point2f>& heads,
                            std::vector<uchar>& status,
                            bool debug=false) const;

    /// Compute the focal length (in px) from image size and dfov
    ///
    /// @param[in] fov  Field of view in radians
    ///
    /// @return         Focal length in pixels
    static double getFocalLength(const cv::Size& img_size, double fov);

    /// Get the yaw, pitch, and roll of the given orientation
    /// Assumes z, y', x'' Tait-Bryan angles
    static void getYPR(const tf2::Quaternion& orientation,
                       double& y,
                       double& p,
                       double& r);

    void processImage(const cv::gpu::GpuMat& image,
                      const cv::gpu::GpuMat& gray_image,
                      const tf2::Quaternion& orientation,
                      const ros::Time& time,
                      double height,
                      bool debug=false) const;

    bool __attribute__((warn_unused_result)) updateFilteredPosition(
            const ros::Time& time,
            const ros::Duration& timeout);

    ////////////////////////
    // INSTANCE VARIABLES //
    ////////////////////////

    const OpticalFlowEstimatorSettings& flow_estimator_settings_;
    const OpticalFlowDebugSettings& debug_settings_;

    bool have_valid_last_image_;
    size_t images_skipped_;

    cv::gpu::GpuMat last_scaled_image_;
    cv::gpu::GpuMat last_scaled_grayscale_image_;
    tf2::Quaternion last_orientation_;

    const ros_utils::SafeTransformWrapper transform_wrapper_;

    double current_altitude_;
    tf2::Quaternion current_orientation_;

    /// Timestamp from last message received
    ros::Time last_message_time_;

    /// Processed image size settings
    cv::Size expected_input_size_;
    cv::Size target_size_;

    /// Publishers
    ros::NodeHandle local_nh_;
    const ros::Publisher debug_average_velocity_vector_image_pub_;
    const ros::Publisher debug_velocity_vector_image_pub_;
    const ros::Publisher correction_pub_;
    const ros::Publisher orientation_pub_;
    const ros::Publisher raw_pub_;
    const ros::Publisher twist_pub_;
};

} // namespace iarc7_vision

#endif // include guard
