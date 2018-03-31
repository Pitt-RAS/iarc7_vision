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

/// Draw specified contour on top of image in given color
void drawContour(cv::Mat& image,
                 const std::vector<cv::Point>& contour,
                 cv::Scalar color);

/// Draw specified rect on top of image in given color
void drawRect(cv::Mat& image,
              const cv::Rect& rect,
              cv::Scalar color);

/// Draw specified rotated rect on top of image in given color
void drawRotatedRect(cv::Mat& image,
                     const cv::RotatedRect& rect,
                     cv::Scalar color);

/// Return true if the pixel (x, y) is inside the image, false otherwise
bool insideImage(const cv::Size& image_size, int x, int y);

/// Return true if the pixel (x, y) inside the rect, false otherwise
bool insideRotatedRect(const cv::RotatedRect& rect, int x, int y);

struct InRangeBuf {
    cv::cuda::GpuMat channels[3];
    cv::cuda::GpuMat inverse;
    cv::cuda::GpuMat buf;
};

/// CUDA implementation of cv::inRange
void inRange(const cv::cuda::GpuMat& src,
             cv::Scalar lowerb,
             cv::Scalar upperb,
             cv::cuda::GpuMat& dst,
             InRangeBuf& buf);

/// See documentation for other inRange
inline void inRange(const cv::cuda::GpuMat& src,
             cv::Scalar lowerb,
             cv::Scalar upperb,
             cv::cuda::GpuMat& dst)
{
    InRangeBuf buf;
    inRange(src, lowerb, upperb, dst, buf);
}

/// Add together all pixels in image that are inside rect
///
/// @param[in]  image  An rgb image
/// @returns           Sum of pixels inside rect
cv::Vec3d sumPatch(const cv::Mat& image, const cv::RotatedRect& rect);

} // end namespace cv_utils

} // end namespace iarc7_vision

#endif // include guard
