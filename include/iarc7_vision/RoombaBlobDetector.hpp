#ifndef _IARC_VISION_ROOMBA_BLOB_DETECTOR_HPP_
#define _IARC_VISION_ROOMBA_BLOB_DETECTOR_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>

#include "iarc7_vision/RoombaEstimatorSettings.hpp"

namespace iarc7_vision
{

class RoombaBlobDetector {
  public:
    RoombaBlobDetector(const RoombaEstimatorSettings& settings,
                       ros::NodeHandle& ph);

    void detect(const cv::cuda::GpuMat& image,
                std::vector<cv::RotatedRect>& bounding_rects);
  private:
    void boundMask(const cv::cuda::GpuMat& mask,
                   std::vector<cv::RotatedRect>& boundRect);
    void checkCorners(const cv::cuda::GpuMat& image,
                      std::vector<cv::RotatedRect>& rects);
    void thresholdFrame(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& dst);

    const RoombaEstimatorSettings& settings_;

    ros::Publisher debug_hsv_slice_pub_;
    ros::Publisher debug_contours_pub_;
    ros::Publisher debug_rects_pub_;
};

}

#endif // include guard
