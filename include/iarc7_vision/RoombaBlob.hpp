#ifndef _IARC_VISION_ROOMBA_BLOB_HPP_
#define _IARC_VISION_ROOMBA_BLOB_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>

namespace iarc7_vision
{

class RoombaBlob {
    void ThresholdFrame(const cv::Mat& image, cv::Mat& dst);
    void BoundMask(const cv::Mat& mask, cv::vector<cv::Rect>& boundRect);
  public:
    void detect(const cv::Mat& image, cv::vector<cv::Rect>& boundRect);
    void DilateBounds(const cv::Mat& image, cv::vector<cv::Rect>& boundRect);
};

}

#endif // include guard
