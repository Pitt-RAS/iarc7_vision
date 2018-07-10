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
    const double a00_;
    const double a01_;
    const double a02_;
    const double a10_;
    const double a11_;
    const double a12_;
    const double a20_;
    const double a21_;
    const double a22_;
    const double offset0_;
    const double offset1_;
    const double offset2_;
    const double gamma_;
    const double final_gamma_;

    cv::Ptr<cv::cuda::LookUpTable> gamma_lut_;
    cv::Ptr<cv::cuda::LookUpTable> final_lut_;

    mutable cv::cuda::GpuMat in_post_gamma_;
    mutable std::array<cv::cuda::GpuMat, 3> in_channels_;
    mutable std::array<cv::cuda::GpuMat, 3> saturated_masks_;
    mutable cv::cuda::GpuMat gbr_order_;
    mutable cv::cuda::GpuMat brg_order_;
    mutable cv::cuda::GpuMat out1_;
    mutable cv::cuda::GpuMat out2_;
    mutable cv::cuda::GpuMat out3_;
    mutable cv::cuda::GpuMat out_float1_;
    mutable cv::cuda::GpuMat out_float2_;
    mutable cv::cuda::GpuMat out_float_;
    mutable cv::cuda::GpuMat out_before_gamma_;
    mutable cv::cuda::GpuMat out_after_gamma_;
    mutable std::array<cv::cuda::GpuMat, 3> out_channels_;

};

} // namespace iarc7_vision

#endif
