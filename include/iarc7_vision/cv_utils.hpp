#ifndef IARC7_VISION_CV_UTILS_HPP_
#define IARC7_VISION_CV_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <vector>

namespace iarc7_vision {

namespace cv_utils {

/// Download a vector from the GPU
void downloadVector(const cv::cuda::GpuMat& mat,
                    std::vector<cv::Point2f>& vector);

/// Download a vector from the GPU
void downloadVector(const cv::cuda::GpuMat& mat,
                    std::vector<uchar>& vector);

/// Draw arrows on top of an image
///
/// @param[in,out] image       The image to draw on
/// @param[in]     tails       Tails of arrows
/// @param[in]     heads       Heads of arrows
/// @param[in]     status      Only draw arrows for which this is nonzero
/// @param[in]     line_color  Color to use for the arrows
void drawArrows(cv::Mat& image,
                const std::vector<cv::Point2f>& tails,
                const std::vector<cv::Point2f>& heads,
                const std::vector<uchar>& status,
                cv::Scalar line_color);

void drawContour(cv::Mat& image,
                 const std::vector<cv::Point>& contour,
                 cv::Scalar color);

void drawRect(cv::Mat& image,
              const cv::Rect& rect,
              cv::Scalar color);

struct InRangeBuf {
    cv::cuda::GpuMat channels[3];
    cv::cuda::GpuMat inverse;
    cv::cuda::GpuMat buf;
};

void inRange(const cv::cuda::GpuMat& src,
             cv::Scalar lowerb,
             cv::Scalar upperb,
             cv::cuda::GpuMat& dst,
             InRangeBuf& buf);

inline void inRange(const cv::cuda::GpuMat& src,
             cv::Scalar lowerb,
             cv::Scalar upperb,
             cv::cuda::GpuMat& dst)
{
    InRangeBuf buf;
    inRange(src, lowerb, upperb, dst, buf);
}

} // end namespace cv_utils

} // end namespace iarc7_vision

#endif // include guard
