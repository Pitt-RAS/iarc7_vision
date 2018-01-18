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
    void detect(const cv::cuda::GpuMat& image,
                const cv::Rect& boundRect,
                cv::Point2f& pos,
                double& angle,
                int camera_canny_threshold);
  private:
    cv::Ptr<cv::GeneralizedHoughGuil> ght_;
    cv::Mat templ_;
};

}

#endif // include guard
