#ifndef IARC7_VISION_COLOR_CORRECTION_MODEL_HPP_
#define IARC7_VISION_COLOR_CORRECTION_MODEL_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>

namespace iarc7_vision {

class ColorCorrectionModel {
  public:
    ColorCorrectionModel(const ros::NodeHandle& nh);
    void correct(const cv::cuda::GpuMat& in,
                 cv::cuda::GpuMat& out,
                 cv::cuda::Stream& stream) const;
  private:
    const double r_a1_;
    const double r_b1_;
    const double r_a2_;
    const double r_b2_;
    const double g_a1_;
    const double g_b1_;
    const double g_a2_;
    const double g_b2_;
    const double b_a1_;
    const double b_b1_;
    const double b_a2_;
    const double b_b2_;
    const double gamma_;

    cv::Ptr<cv::cuda::LookUpTable> lut_;
};

} // namespace iarc7_vision

#endif
