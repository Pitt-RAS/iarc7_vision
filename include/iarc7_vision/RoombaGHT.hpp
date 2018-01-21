#ifndef _IARC_VISION_ROOMBA_GHT_HPP_
#define _IARC_VISION_ROOMBA_GHT_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ros/ros.h>

#include "iarc7_vision/RoombaEstimatorSettings.hpp"

namespace iarc7_vision
{

class RoombaGHT {
  public:
    RoombaGHT(const RoombaEstimatorSettings& settings,
              ros::NodeHandle& private_nh);

    bool detect(const cv::cuda::GpuMat& image,
                const cv::Rect& bounding_rect,
                cv::Point2f& pos,
                double& angle);
    void onSettingsChanged();
  private:
    ros::NodeHandle& private_nh_;
    cv::Ptr<cv::GeneralizedHoughBallard> ght_;
    const RoombaEstimatorSettings& settings_;
    ros::Publisher debug_edges_pub_;
};

} // namespace iarc7_vision

#endif // include guard
