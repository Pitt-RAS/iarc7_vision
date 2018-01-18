#ifndef _IARC_VISION_ROOMBA_GHT_HPP_
#define _IARC_VISION_ROOMBA_GHT_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ros/ros.h>

namespace iarc7_vision
{

class RoombaGHT {
  public:
    void setup(float pixels_per_meter,
               float roomba_plate_width,
               int ght_levels,
               int ght_dp,
               int votes_threshold,
               int template_canny_threshold);
    float detect(const cv::Mat& image, cv::Rect& boundRect, cv::Point2f& pos, 
                 int camera_canny_threshold);
    float detect(const cv::gpu::GpuMat& image, cv::Rect& boundRect,
                 cv::Point2f& pos, int camera_canny_threshold);
  private:
    bool useGpu = false;
    cv::Ptr<cv::GeneralizedHough> ght;
    cv::Mat templ;
    cv::Ptr<cv::gpu::GeneralizedHough_GPU> gpu_ght;
    cv::gpu::GpuMat gpu_templ;
};

}

#endif // include guard
