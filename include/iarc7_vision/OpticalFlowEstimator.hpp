#ifndef IARC7_VISION_OPTICAL_FLOW_ESTIMATOR_HPP_
#define IARC7_VISION_OPTICAL_FLOW_ESTIMATOR_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
    double camera_vertical_threshold;
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
    bool debug_average_vector_image;
    bool debug_intermediate_velocities;
    bool debug_orientation;
    bool debug_times;
    bool debug_vectors_image;
};

class OpticalFlowEstimator {
  public:

    //////////////////
    // CONSTRUCTORS //
    //////////////////

    OpticalFlowEstimator(
            const OpticalFlowEstimatorSettings& flow_estimator_settings,
            const OpticalFlowDebugSettings& debug_settings,
            const std::string& expected_image_format);

    ////////////////////
    // PUBLIC METHODS //
    ////////////////////

    /// MUST be called when either of the settings objects passed into the
    /// constructor have their variables changed
    bool __attribute__((warn_unused_result)) onSettingsChanged();

    /// Process a new image message
    void update(const sensor_msgs::Image::ConstPtr& message);

    /// MUST be called successfully before `update` is called
    bool __attribute__((warn_unused_result)) waitUntilReady(
            const ros::Duration& startup_timeout);

  private:

    /////////////////////
    // PRIVATE METHODS //
    /////////////////////

    /// Decides whether the drone is in a position where it can make an optical
    /// flow estimate
    ///
    /// @returns  True if a flow estimate can be made, false otherwise
    bool canEstimateFlow() const;

    /// Calculate new velocity estimate
    ///
    /// @param[in]  average_vec {Movement of the average feature between the
    ///                          previous frame and the current frame}
    /// @param[in]  time        Timestamp of the image
    ///
    /// @return                 Velocity estimate
    geometry_msgs::TwistWithCovarianceStamped estimateVelocityFromFlowVector(
            const cv::Point2f& average_vec,
            const ros::Time& time) const;

    /// Finds the average of the given feature vectors, filtering out points
    /// outside of the cutoff region
    ///
    /// Each feature vector is the movement of the feature in the camera frame
    /// from the previous frame to the current frame
    ///
    /// @param[in]  tails      Tails of the input vectors
    /// @param[in]  heads      Heads of the input vectors
    /// @param[in]  status     Status of each vector, 1 if valid and 0 otherwise
    /// @param[in]  x_cutoff   {Width of region to be discarded on each side as a
    ///                         percentage of the total width}
    /// @param[in]  y_cutoff   {Height of region to be discarded on each side as
    ///                         a percentage of the total height}
    /// @param[in]  image_size {Size of image that the points are from, used for
    ///                         calculating cutoffs}
    /// @param[out] average    Average movement of the features in the frame
    ///
    /// @return                {True if result is valid (i.e. at least one
    ///                         valid point)}
    static bool findAverageVector(const std::vector<cv::Point2f>& tails,
                                         const std::vector<cv::Point2f>& heads,
                                         const std::vector<uchar>& status,
                                         const double x_cutoff,
                                         const double y_cutoff,
                                         const cv::Size& image_size,
                                         cv::Point2f& average);

    /// Process the given current and last frames to find flow vectors
    ///
    /// @param[in]  curr_frame       The current frame, in RGB8
    /// @param[in]  curr_gray_frame  The current frame, in grayscale
    /// @param[in]  last_frame       The last frame, in RGB8
    /// @param[in]  last_gray_frame  The last frame, in grayscale
    /// @param[out] tails            The tails of the flow vectors
    /// @param[out] heads            The heads of the flow vectors
    /// @param[out] status           {The status of each flow vector, nonzero
    ///                               for valid}
    /// @param[in]  debug            {Whether to spit out debug info, like
    ///                               images from intermediate steps or with
    ///                               arrows drawn}
    void findFeatureVectors(const cv::cuda::GpuMat& curr_frame,
                            const cv::cuda::GpuMat& curr_gray_frame,
                            const cv::cuda::GpuMat& last_frame,
                            const cv::cuda::GpuMat& last_gray_frame,
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

    /// Do all necessary processing on the new image, and publish results to
    /// relevant topics
    ///
    /// @param[in] image        Current frame to process, in RGB8
    /// @param[in] gray_image   Current frame to process, in MONO8
    /// @param[in] time         Timestamp when `image` was captured
    /// @param[in] debug        Whether to spit out messages on debug topics
    void processImage(const cv::cuda::GpuMat& image,
                      const cv::cuda::GpuMat& gray_image,
                      const ros::Time& time,
                      bool debug=false) const;

    /// Resize image and convert to grayscale
    ///
    /// All three of the inputs should be different images (or either of the
    /// outputs can be empty)
    ///
    /// @param[in]  image   Image to resize
    /// @param[out] scaled  Image resized to target_size_
    /// @param[out] gray    Grayscale image resized to target_size_
    void resizeAndConvertImages(const cv::cuda::GpuMat& image,
                                cv::cuda::GpuMat& scaled,
                                cv::cuda::GpuMat& gray) const;

    /// Update altitude measurement and camera transform
    ///
    /// @param[in] time    {Time of latest measurements after function returns
    ///                     successfully}
    /// @param[in] timeout Max time to wait for transforms
    ///
    /// @return            True if transform was updated successfully
    bool __attribute__((warn_unused_result)) updateFilteredPosition(
            const ros::Time& time,
            const ros::Duration& timeout);

    ////////////////////////
    // INSTANCE VARIABLES //
    ////////////////////////

    uint32_t grayscale_conversion_constant_;
    std::string image_encoding_;

    const OpticalFlowEstimatorSettings& flow_estimator_settings_;
    const OpticalFlowDebugSettings& debug_settings_;

    cv::Ptr<cv::cuda::CornersDetector> gpu_features_detector_;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> gpu_d_pyrLK_;

    bool have_valid_last_image_;
    size_t images_skipped_;

    cv::cuda::GpuMat last_scaled_image_;
    cv::cuda::GpuMat last_scaled_grayscale_image_;

    const ros_utils::SafeTransformWrapper transform_wrapper_;

    double current_altitude_;
    tf2::Quaternion current_orientation_;
    tf2::Quaternion last_orientation_;
    geometry_msgs::TransformStamped current_camera_to_level_quad_tf_;
    geometry_msgs::TransformStamped last_camera_to_level_quad_tf_;

    /// Timestamp from last message received
    ros::Time last_message_time_;

    /// Processed image size settings
    cv::Size expected_input_size_;
    cv::Size target_size_;

    /// Publishers
    ros::NodeHandle local_nh_;
    const ros::Publisher debug_average_velocity_vector_image_pub_;
    const ros::Publisher debug_level_quad_raw_pub_;
    const ros::Publisher debug_camera_rel_raw_pub_;
    const ros::Publisher debug_correction_pub_;
    const ros::Publisher debug_raw_pub_;
    const ros::Publisher debug_unrotated_vel_pub_;
    const ros::Publisher debug_velocity_vector_image_pub_;
    const ros::Publisher orientation_pub_;
    const ros::Publisher twist_pub_;
};

} // namespace iarc7_vision

#endif // include guard
