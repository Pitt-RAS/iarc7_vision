#ifndef _IARC_VISION_ROOMBA_BLOB_DETECTOR_HPP_
#define _IARC_VISION_ROOMBA_BLOB_DETECTOR_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>

namespace iarc7_vision
{

class RoombaBlobDetector {
    void boundMask(const cv::cuda::GpuMat& mask,
                   std::vector<cv::Rect>& boundRect);
    void dilateBounds(const cv::cuda::GpuMat& image,
                      std::vector<cv::Rect>& boundRect);
    void thresholdFrame(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& dst);
  public:
    void detect(const cv::cuda::GpuMat& image,
                std::vector<cv::Rect>& boundRect);
};

}

#endif // include guard
