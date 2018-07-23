#ifndef IARC7_VISION_UNDISTORTION_MODEL_HPP_
#define IARC7_VISION_UNDISTORTION_MODEL_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>

namespace iarc7_vision {

class UndistortionModel {
  public:
    UndistortionModel(const ros::NodeHandle& nh,
                      const cv::Size& image_size);

    void undistort(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& out, cv::cuda::Stream& stream) const;

    cv::Size getUndistortedSize() const { return new_image_size_; }

  private:
    const cv::Size image_size_;
    const cv::Size new_image_size_;

    cv::cuda::GpuMat map1_;
    cv::cuda::GpuMat map2_;
};

} // namespace iarc7_vision

#endif
