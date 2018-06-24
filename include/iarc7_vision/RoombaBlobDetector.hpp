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

/// Takes images, finds roombas, and outputs detections in image coordinates
///
/// Uses an HSV slice and morphology for finding possible roombas
class RoombaBlobDetector {
  public:
    RoombaBlobDetector(const RoombaEstimatorSettings& settings,
                       ros::NodeHandle& ph);

    /// Processes current frame
    ///
    /// @param[in]   image           Current frame to process (in rgb8)
    /// @param[out]  bounding_rects  Bounding rectangles of detected top plates
    void detect(const cv::cuda::GpuMat& image,
                std::vector<cv::RotatedRect>& bounding_rects,
                std::vector<double>& flip_certainties);
  private:

    /// Find roomba rotated bounding rects in mask
    void boundMask(const cv::cuda::GpuMat& mask,
                   std::vector<cv::RotatedRect>& boundRect);

    /// Examine four corners of each detection rect.  Based on which corners
    /// are white, rotate rect 180 degrees to point in the correct direction.
    ///
    /// @param[in]      image  Image to examine
    /// @param[in,out]  rects  Detection rects for roombas in image
    void checkCorners(const cv::cuda::GpuMat& image,
                      std::vector<cv::RotatedRect>& rects,
                      std::vector<double>& flip_certainties);

    /// Perform HSV slice and morphology to get pixels which are likely
    /// to be roomba top plates
    ///
    /// @param[in]   image  rgb8 input image
    /// @param[out]  dst    mono8 output mask, nonzero pixels are top plates
    void thresholdFrame(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& dst);

    const RoombaEstimatorSettings& settings_;

    ros::Publisher debug_hsv_slice_pub_;
    ros::Publisher debug_contours_pub_;
    ros::Publisher debug_rects_pub_;
};

}

#endif // include guard
