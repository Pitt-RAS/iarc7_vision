#include "iarc7_vision/cv_utils.hpp"

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

} // end namespace cv_utils

} // end namespace iarc7_vision
