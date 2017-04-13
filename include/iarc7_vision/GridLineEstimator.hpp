#ifndef IARC7_VISION_GRIDLINE_ESTIMATOR_HPP_
#define IARC7_VISION_GRIDLINE_ESTIMATOR_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros_utils/SafeTransformWrapper.hpp>
#include <tf2_ros/transform_listener.h>

namespace iarc7_vision {

struct LineExtractorSettings {
    double canny_low_threshold;
    double canny_high_threshold;
    int canny_sobel_size;
    double hough_rho_resolution;
    double hough_theta_resolution;
    double hough_thresh_fraction;
};

struct GridLineDebugSettings {
    bool debug_edges;
    bool debug_lines;
};

class GridLineEstimator {
  public:
    GridLineEstimator(double fov,
                      const LineExtractorSettings& line_estimator_settings,
                      const GridLineDebugSettings& debug_settings);
    void update(const cv::Mat& image, const ros::Time& time);

  private:
    /// Compute the focal length (in px) from image size and dfov
    ///
    /// @param[in] fov Field of view in radians
    static double getFocalLength(const cv::Size& img_size, double fov);

    /// Return all the lines in the image
    ///
    /// @param[out] lines {Lines in the form (rho, theta), where theta is the
    ///                    angle (from right, positive is down) and rho is the
    //                     distance (in pixels) from the center of the image
    //                     (NOTE: rho is not the same as in OpenCV HoughLines)}
    /// @param[in]  image Input image
    /// @param[in] height Approximate altitude of the camera, in meters
    ///
    /// TODO: change units of height parameter to pixels
    void getLines(std::vector<cv::Vec2f>& lines,
                  const cv::Mat& image,
                  double height);

    cv::Mat image_sized_;
    cv::Mat image_hsv_;
    cv::Mat image_edges_;
    cv::Mat image_hsv_channels_[3];

    cv::gpu::HoughLinesBuf gpu_hough_buf_;
    cv::gpu::GpuMat gpu_image_;
    cv::gpu::GpuMat gpu_image_sized_;
    cv::gpu::GpuMat gpu_image_hsv_;
    cv::gpu::GpuMat gpu_image_edges_;
    cv::gpu::GpuMat gpu_lines_;
    cv::gpu::GpuMat gpu_image_hsv_channels_[3];

    const double fov_; // diagonal field of view of our bottom camera

    const LineExtractorSettings& line_extractor_settings_;

    const GridLineDebugSettings& debug_settings_;
    ros::Publisher debug_edges_pub_;
    ros::Publisher debug_lines_pub_;

    ros_utils::SafeTransformWrapper transform_wrapper_;
};

} // namespace iarc7_vision

#endif // include guard
