#include "iarc7_vision/cv_utils.hpp"

#include <opencv2/cudaarithm.hpp>
#include <ros/ros.h>

namespace iarc7_vision {

namespace cv_utils {

void downloadVector(const cv::cuda::GpuMat& mat,
                    std::vector<cv::Point2f>& vector)
{
    vector.resize(mat.cols);
    cv::Mat cpu_mat(1, mat.cols, CV_32FC2, (void*)&vector[0]);
    mat.download(cpu_mat);
}

void downloadVector(const cv::cuda::GpuMat& mat,
                    std::vector<uchar>& vector)
{
    vector.resize(mat.cols);
    cv::Mat cpu_mat(1, mat.cols, CV_8UC1, (void*)&vector[0]);
    mat.download(cpu_mat);
}

void drawArrows(cv::Mat& image,
                const std::vector<cv::Point2f>& tails,
                const std::vector<cv::Point2f>& heads,
                const std::vector<uchar>& status,
                cv::Scalar line_color)
{
    for (size_t i = 0; i < tails.size(); ++i) {
        if (status[i]) {
            int line_thickness = 1;

            cv::Point p = tails[i];
            cv::Point q = heads[i];

            double angle = std::atan2(p.y - q.y, p.x - q.x);
            double hypotenuse = std::hypot(p.y - q.y, p.x - q.x);

            // Here we lengthen the arrow by a factor of three.
            q.x = p.x - 3 * hypotenuse * std::cos(angle);
            q.y = p.y - 3 * hypotenuse * std::sin(angle);

            // Now we draw the main line of the arrow.
            cv::line(image, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = q.x + 9 * std::cos(angle + CV_PI / 4);
            p.y = q.y + 9 * std::sin(angle + CV_PI / 4);
            cv::line(image, p, q, line_color, line_thickness);

            p.x = q.x + 9 * cos(angle - CV_PI / 4);
            p.y = q.y + 9 * sin(angle - CV_PI / 4);
            cv::line(image, p, q, line_color, line_thickness);
        }
    }
}

void drawContour(cv::Mat& image,
                 const std::vector<cv::Point>& contour,
                 cv::Scalar color)
{
    for (size_t i = 0; i < contour.size() - 1; i++) {
        cv::line(image, contour[i], contour[i + 1], color);
    }
    cv::line(image, contour[0], contour.back(), color);
}

void drawRect(cv::Mat& image,
              const cv::Rect& rect,
              cv::Scalar color)
{
    cv::Point p1 = rect.tl();
    cv::Point p2 = rect.tl();

    p2.x += rect.width;
    cv::line(image, p1, p2, color);

    p1 = rect.br();
    cv::line(image, p1, p2, color);

    p2 = rect.tl();
    p2.y += rect.height;
    cv::line(image, p1, p2, color);

    p1 = rect.tl();
    cv::line(image, p1, p2, color);
}

void drawRotatedRect(cv::Mat& image,
                     const cv::RotatedRect& rect,
                     cv::Scalar color)
{
    cv::Point2f pts[4];
    rect.points(pts);

    cv::line(image, pts[0], pts[1], color, 3);
    cv::line(image, pts[1], pts[2], color, 3);
    cv::line(image, pts[2], pts[3], color, 3);
    cv::line(image, pts[3], pts[0], color, 3);
}

bool insideImage(const cv::Size& image_size, int x, int y)
{
    return (x >= 0)
        && (y >= 0)
        && (x < image_size.width)
        && (y < image_size.height);
}

bool insideRotatedRect(const cv::RotatedRect& rect, int x, int y)
{
    float rads = rect.angle * M_PI / 180;
    cv::Vec2f dirx ( std::cos(rads), std::sin(rads));
    cv::Vec2f diry (-std::sin(rads), std::cos(rads));

    cv::Vec2f offset = cv::Point2f(x, y) - rect.center;
    if (std::abs(dirx.dot(offset)) <= rect.size.width  / 2
     && std::abs(diry.dot(offset)) <= rect.size.height / 2) {
        return true;
    } else {
        return false;
    }
}

void inRange(const cv::cuda::GpuMat& src,
             cv::Scalar lowerb,
             cv::Scalar upperb,
             cv::cuda::GpuMat& dst,
             InRangeBuf& buf)
{
    ROS_ASSERT(src.type() == CV_8UC3 || src.type() == CV_8UC4);

    cv::cuda::split(src, buf.channels);

    cv::cuda::subtract(cv::Scalar(255), buf.channels[0], buf.inverse);
    cv::cuda::threshold(buf.inverse,
                        buf.buf,
                        255 - lowerb[0],
                        255,
                        cv::THRESH_BINARY_INV);
    cv::cuda::bitwise_and(buf.buf, buf.buf, dst);
    for (int i = 1; i < 3; i++) {
        cv::cuda::subtract(cv::Scalar(255), buf.channels[i], buf.inverse);
        cv::cuda::threshold(buf.inverse,
                            buf.buf,
                            255 - lowerb[i],
                            255,
                            cv::THRESH_BINARY_INV);
        cv::cuda::bitwise_and(buf.buf, dst, dst);
    }

    for (int i = 0; i < 3; i++) {
        cv::cuda::threshold(buf.channels[i],
                            buf.buf,
                            upperb[i],
                            255,
                            cv::THRESH_BINARY_INV);
        cv::cuda::bitwise_and(buf.buf, dst, dst);
    }
}

cv::Vec3d sumPatch(const cv::Mat& image, const cv::RotatedRect& rect)
{
    cv::Rect bounding_rect = rect.boundingRect2f();
    size_t count = 0;
    cv::Vec4d total (0, 0, 0);
    for (int y = bounding_rect.y; y <= bounding_rect.y + bounding_rect.width; y++) {
        for (int x = bounding_rect.x; x <= bounding_rect.x + bounding_rect.width; x++) {
            if (cv_utils::insideImage(image.size(), x, y)
             && cv_utils::insideRotatedRect(rect, x, y)) {
                count++;
                total += image.at<cv::Vec4b>(y, x);
            }
        }
    }

    return cv::Vec3d(total[0], total[1], total[2]) / double(count);
}

} // end namespace cv_utils

} // end namespace iarc7_vision
