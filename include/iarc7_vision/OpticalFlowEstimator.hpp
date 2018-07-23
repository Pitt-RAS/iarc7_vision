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

#include "iarc7_vision/RoombaImageLocation.hpp"

#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

namespace iarc7_vision {

enum class VectorFilterType
{
    Average,
    Median,
    Statistical
};

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
    bool crop;
    int crop_width;
    int crop_height;
    double variance;
    double variance_scale;
    double x_cutoff_region_velocity_measurement;
    double y_cutoff_region_velocity_measurement;
    int debug_frameskip;
    double tf_timeout;
    double max_rotational_vel;
    VectorFilterType vector_filter;
    int min_vectors;
    double max_filtered_variance;
    double max_normalized_element_variance;
    double hist_scale_factor;
    double hist_image_size_scale;

    void set_vector_filter_str(const std::string& str) {
        if (str == "statistical") vector_filter = VectorFilterType::Statistical;
        else if (str == "median") vector_filter = VectorFilterType::Median;
        else if (str == "average") vector_filter = VectorFilterType::Average;
        else ROS_ERROR("Invalid vector filter type set, defaulting to last set");
    }

    void get_vector_filter_str(std::string& str) {
        switch (vector_filter)
        {
            case VectorFilterType::Statistical:
                str = "statistical";
                break;
            case VectorFilterType::Median:
                str = "median";
                break;
            case VectorFilterType::Average:
                str = "average";
        }
    }
};

struct OpticalFlowDebugSettings {
    bool debug_average_vector_image;
    bool debug_intermediate_velocities;
    bool debug_orientation;
    bool debug_times;
    bool debug_vectors_image;
    bool debug_hist;
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
    void update(const cv::cuda::GpuMat& curr_image,
                const ros::Time& time,
                const std::vector<RoombaImageLocation>& roomba_image_locations,
                const bool images_skipped);

    /// MUST be called successfully before `update` is called
    bool __attribute__((warn_unused_result)) waitUntilReady(
            const ros::Duration& startup_timeout);

  private:

    /////////////////////
    // PRIVATE METHODS //
    /////////////////////

    /// Calculates pitch and roll rates, averaged between last frame time and
    /// this frame time
    void calculateRotationRate(const ros::Time& time,
                               double& dpitch_dt,
                               double& droll_dt) const;

    /// Decides whether the drone is in a position where it can make an optical
    /// flow estimate
    ///
    /// @returns  True if a flow estimate can be made, false otherwise
    bool canEstimateFlow(const ros::Time& time) const;

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
    /// @param[in] roomba_image_locations {Vector of roomba image locations
    ///                                   in resolution indpenedent units}
    /// @param[in]  image_size {Size of image that the points are from, used for
    ///                         calculating cutoffs}
    /// @param[in]  curr_frame The current frame, in RGB8
    /// @param[in]  time       Timestamp when `image` was captured
    /// @param[in]  debug      {Whether to spit out debug info, like
    ///                         images from intermediate steps or with
    ///                         arrows drawn}
    /// @param[out] average    Average movement of the features in the frame
    ///
    /// @return                {True if result is valid (i.e. at least one
    ///                         valid point)}
    bool findAverageVector(const std::vector<cv::Point2f>& tails,
                                         const std::vector<cv::Point2f>& heads,
                                         const std::vector<uchar>& status,
                                         const double x_cutoff,
                                         const double y_cutoff,
                                         const std::vector<RoombaImageLocation>&
                                             roomba_image_locations,
                                         const cv::Size& image_size,
                                         const cv::cuda::GpuMat& curr_frame,
                                         const ros::Time& time,
                                         const bool debug,
                                         cv::Point2f& average) const;

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
    /// @param[in] roomba_image_locations {Vector of roomba image locations
    ///                                   in resolution indpenedent units}
    /// @param[in] debug        Whether to spit out messages on debug topics
    void processImage(const cv::cuda::GpuMat& image,
                      const cv::cuda::GpuMat& gray_image,
                      const ros::Time& time,
                      const std::vector<RoombaImageLocation>&
                          roomba_image_locations,
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
    double fov_;

    /// Publishers
    ros::NodeHandle local_nh_;
    const ros::Publisher debug_orientation_rate_pub_;
    const ros::Publisher debug_average_velocity_vector_image_pub_;
    const ros::Publisher debug_level_quad_raw_pub_;
    const ros::Publisher debug_camera_rel_raw_pub_;
    const ros::Publisher debug_correction_pub_;
    const ros::Publisher debug_hist_pub_;
    const ros::Publisher debug_raw_pub_;
    const ros::Publisher debug_unrotated_vel_pub_;
    const ros::Publisher debug_velocity_vector_image_pub_;
    const ros::Publisher debug_filtered_velocity_vector_image_pub_;
    const ros::Publisher debug_flow_quality_pub_;
    const ros::Publisher orientation_pub_;
    const ros::Publisher twist_pub_;
};

} // namespace iarc7_vision

#endif // include guard
