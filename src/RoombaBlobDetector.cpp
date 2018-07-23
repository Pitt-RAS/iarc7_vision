#include <chrono>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include "iarc7_vision/cv_utils.hpp"
#include "iarc7_vision/RoombaBlobDetector.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Dense>
#pragma GCC diagnostic pop

namespace iarc7_vision
{

RoombaBlobDetector::RoombaBlobDetector(const RoombaEstimatorSettings& settings,
                                       ros::NodeHandle& ph,
                                       const cv::Size& image_size)
    : settings_(settings),
      image_size_(image_size),
      structuring_element_(cv::getStructuringElement(
              cv::MORPH_RECT,
              cv::Size(settings_.morphology_size, settings_.morphology_size))),
      morphology_open_(cv::cuda::createMorphologyFilter(
              cv::MORPH_OPEN,
              CV_8UC1,
              structuring_element_,
              cv::Point(-1, -1),
              settings_.morphology_iterations)),
      morphology_close_(cv::cuda::createMorphologyFilter(
              cv::MORPH_CLOSE,
              CV_8UC1,
              structuring_element_,
              cv::Point(-1, -1),
              settings_.morphology_iterations))
{
    if (settings_.debug_hsv_slice) {
        debug_hsv_slice_pub_ = ph.advertise<sensor_msgs::Image>("hsv_slice",
                                                                10);
    }

    if (settings_.debug_contours) {
        debug_contours_pub_ = ph.advertise<sensor_msgs::Image>("contours",
                                                               10);
    }
}

void RoombaBlobDetector::thresholdFrame(const cv::cuda::GpuMat& image,
                                        cv::cuda::GpuMat& dst) const
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    cv::cuda::cvtColor(image, hsv_image_, cv::COLOR_RGB2HSV);
    cv::cuda::split(hsv_image_, hsv_channels_.data());

    dst.create(image.rows, image.cols, CV_8U);
    dst.setTo(cv::Scalar(0, 0, 0, 0));

    // Green slice
    cv_utils::inRange(hsv_image_,
                      cv::Scalar(settings_.hsv_slice_h_green_min,
                                 settings_.hsv_slice_s_min,
                                 settings_.hsv_slice_v_min),
                      cv::Scalar(settings_.hsv_slice_h_green_max,
                                 settings_.hsv_slice_s_max,
                                 settings_.hsv_slice_v_max),
                      range_mask_);
    cv::cuda::bitwise_or(dst, range_mask_, dst);
    // Upper red slice
    cv_utils::inRange(hsv_image_,
                      cv::Scalar(settings_.hsv_slice_h_red1_min,
                                 settings_.hsv_slice_s_min,
                                 settings_.hsv_slice_v_min),
                      cv::Scalar(settings_.hsv_slice_h_red1_max,
                                 settings_.hsv_slice_s_max,
                                 settings_.hsv_slice_v_max),
                      range_mask_);
    cv::cuda::bitwise_or(dst, range_mask_, dst);
    // Lower red slice
    cv_utils::inRange(hsv_image_,
                      cv::Scalar(settings_.hsv_slice_h_red2_min,
                                 settings_.hsv_slice_s_min,
                                 settings_.hsv_slice_v_min),
                      cv::Scalar(settings_.hsv_slice_h_red2_max,
                                 settings_.hsv_slice_s_max,
                                 settings_.hsv_slice_v_max),
                      range_mask_);
    cv::cuda::bitwise_or(dst, range_mask_, dst);

    const auto slice_time = std::chrono::high_resolution_clock::now();

    morphology_open_->apply(dst, dst);
    morphology_close_->apply(dst, dst);

    const auto morph_time = std::chrono::high_resolution_clock::now();

    const auto count = [](const auto& a, const auto& b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                b - a).count();
    };
    ROS_DEBUG_STREAM(
            "Slice: " << count(start_time, slice_time) << std::endl
         << "Morph: " << count(slice_time, morph_time) << std::endl);

    ROS_ASSERT(dst.channels() == 1);
}

void RoombaBlobDetector::boundMask(
        const cv::cuda::GpuMat& mask,
        std::vector<cv::RotatedRect>& boundRect) const
{
    cv::Mat mask_cpu;
    mask.download(mask_cpu);

    std::vector<std::vector<cv::Point>> contours;

    //////////////////////////////////////////////////////////////////////////
    /// Find bounding contours of blobs in mask
    //////////////////////////////////////////////////////////////////////////

    // findContours does not exist for the gpu
    cv::findContours(mask_cpu,
                     contours,
                     CV_RETR_EXTERNAL,
                     CV_CHAIN_APPROX_SIMPLE);

    if (settings_.debug_contours) {
        cv::Mat contour_image = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC3);

        for (const auto& contour : contours) {
            cv_utils::drawContour(contour_image,
                                  contour,
                                  cv::Scalar(255, 255, 255));
        }

        const cv_bridge::CvImage cv_image {
            std_msgs::Header(),
            sensor_msgs::image_encodings::RGB8,
            contour_image
        };

        debug_contours_pub_.publish(cv_image.toImageMsg());
    }

    //////////////////////////////////////////////////////////////////////////
    /// Get orientation for each found contour
    //////////////////////////////////////////////////////////////////////////

    boundRect.resize(0); // Clear the vector
    cv::RotatedRect rect;
    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::Moments moments = cv::moments(contours[i]);

        // throw out blobs that are too small or too large
        if (moments.m00 < settings_.min_roomba_blob_size
         || moments.m00 > settings_.max_roomba_blob_size) {
            ROS_DEBUG("Rejecting bad blob size: %f", moments.m00);
            continue;
        }

        // Calculate eigenvectors of covariance of blob to get orientation
        Eigen::Matrix2d covariance;
        covariance(0, 0) = moments.nu20;
        covariance(1, 0) = moments.nu11;
        covariance(0, 1) = moments.nu11;
        covariance(1, 1) = moments.nu02;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver (covariance);
        Eigen::Vector2d evector0 = eigensolver.eigenvectors().col(0);
        Eigen::Vector2d evector1 = eigensolver.eigenvectors().col(1);

        // Calculate bounding rect for the calculated orientation
        bool have_point = false;
        double rect_max_x = 0;
        double rect_min_x = 0;
        double rect_max_y = 0;
        double rect_min_y = 0;
        for (const cv::Point& point : contours[i]) {
            Eigen::Vector2d p;
            p(0) = point.x;
            p(1) = point.y;

            double dx = evector0.dot(p);
            double dy = evector1.dot(p);

            if (!have_point || dx < rect_min_x) {
                rect_min_x = dx;
            }

            if (!have_point || dx > rect_max_x) {
                rect_max_x = dx;
            }

            if (!have_point || dy < rect_min_y) {
                rect_min_y = dy;
            }

            if (!have_point || dy > rect_max_y) {
                rect_max_y = dy;
            }

            have_point = true;
        }

        if (!have_point) {
            throw ros::Exception("Empty contour");
        }

        // Fill out results
        Eigen::Vector2d center = evector0 * ((rect_max_x + rect_min_x) / 2)
                               + evector1 * ((rect_max_y + rect_min_y) / 2);

        rect = cv::RotatedRect(
                cv::Point2f(center(0), center(1)),
                cv::Size2f(rect_max_x - rect_min_x, rect_max_y - rect_min_y),
                -std::atan2(evector1(0), evector1(1)) * 180 / M_PI);

        boundRect.push_back(rect);
    }
}

void RoombaBlobDetector::checkCorners(
        const cv::cuda::GpuMat& image,
        std::vector<cv::RotatedRect>& rects,
        std::vector<double>& flip_certainties) const
{
    ROS_ASSERT(flip_certainties.empty());
    flip_certainties.reserve(rects.size());

    cv::Mat cpu_image;
    image.download(cpu_image);

    float scale = 0.2;

    for (cv::RotatedRect& rect : rects) {
        cv::RotatedRect window (cv::Point2f(), rect.size * scale, rect.angle);

        float rads = rect.angle * M_PI / 180;
        cv::Point2f offset_x = rect.size.width
                             * (1 - scale) / 2
                             * cv::Point2f(std::cos(rads), std::sin(rads));
        cv::Point2f offset_y = rect.size.height
                             * (1 - scale) / 2
                             * cv::Point2f(-std::sin(rads), std::cos(rads));

        // True if the patch is red/green, false otherwise
        cv::Mat corners = cv::Mat::zeros(2, 2, CV_8UC1);

        for (int i = -1; i != 3; i += 2) {
            for (int j = -1; j != 3; j += 2) {
                window.center = rect.center + i*offset_x + j*offset_y;
                cv::Vec3d patch_sum = cv_utils::sumPatch(cpu_image, window);

                cv::Mat patch_sum_mat = cv::Mat::zeros(1, 1, CV_8UC3);
                patch_sum_mat.at<cv::Vec3b>(0, 0) = patch_sum;
                cv::cvtColor(patch_sum_mat, patch_sum_mat, cv::COLOR_RGB2HSV);

                // True if the patch is red/green, false otherwise
                cv::Mat patch_good_mat = cv::Mat::zeros(1, 1, CV_8U);
                patch_good_mat.at<uchar>(0, 0) = 0;

                cv::Mat range_mask;

                // Green slice
                cv::inRange(patch_sum_mat,
                            cv::Scalar(settings_.hsv_slice_h_green_min,
                                       settings_.hsv_slice_s_min,
                                       settings_.hsv_slice_v_min),
                            cv::Scalar(settings_.hsv_slice_h_green_max,
                                       settings_.hsv_slice_s_max,
                                       settings_.hsv_slice_v_max),
                            range_mask);
                cv::bitwise_or(patch_good_mat, range_mask, patch_good_mat);
                // Upper red slice
                cv::inRange(patch_sum_mat,
                            cv::Scalar(settings_.hsv_slice_h_red1_min,
                                       settings_.hsv_slice_s_min,
                                       settings_.hsv_slice_v_min),
                            cv::Scalar(settings_.hsv_slice_h_red1_max,
                                       settings_.hsv_slice_s_max,
                                       settings_.hsv_slice_v_max),
                            range_mask);
                cv::bitwise_or(patch_good_mat, range_mask, patch_good_mat);
                // Lower red slice
                cv::inRange(patch_sum_mat,
                            cv::Scalar(settings_.hsv_slice_h_red2_min,
                                       settings_.hsv_slice_s_min,
                                       settings_.hsv_slice_v_min),
                            cv::Scalar(settings_.hsv_slice_h_red2_max,
                                       settings_.hsv_slice_s_max,
                                       settings_.hsv_slice_v_max),
                            range_mask);
                cv::bitwise_or(patch_good_mat, range_mask, patch_good_mat);

                corners.at<uchar>((i+1)/2, (j+1)/2)
                        = patch_good_mat.at<uchar>(0, 0);
            }
        }

        if (!corners.at<uchar>(0, 0)
         && !corners.at<uchar>(0, 1)
         && corners.at<uchar>(1, 0)
         && corners.at<uchar>(1, 1)) {
            rect.angle += 180;
            rect.angle = std::fmod(rect.angle, 360);
            flip_certainties.push_back(1.0);
        } else if (corners.at<uchar>(0, 0)
                && corners.at<uchar>(0, 1)
                && !corners.at<uchar>(1, 0)
                && !corners.at<uchar>(1, 1)) {
            flip_certainties.push_back(1.0);
        } else {
            flip_certainties.push_back(0.0);
        }
    }
}

void RoombaBlobDetector::detect(const cv::cuda::GpuMat& image,
                                std::vector<cv::RotatedRect>& bounding_rects,
                                std::vector<double>& flip_certainties) const
{
    ROS_ASSERT(image.size() == image_size_);

    const auto start_time = std::chrono::high_resolution_clock::now();

    cv::cuda::GpuMat mask;
    thresholdFrame(image, mask);

    const auto threshold_time = std::chrono::high_resolution_clock::now();

    if (settings_.debug_hsv_slice) {
        cv::Mat mask_cpu;
        mask.download(mask_cpu);

        const cv_bridge::CvImage cv_image {
            std_msgs::Header(),
            sensor_msgs::image_encodings::MONO8,
            mask_cpu
        };

        debug_hsv_slice_pub_.publish(cv_image.toImageMsg());
    }

    boundMask(mask, bounding_rects);
    checkCorners(image, bounding_rects, flip_certainties);

    const auto final_time = std::chrono::high_resolution_clock::now();

    const auto count = [](const auto& a, const auto& b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                b - a).count();
    };
    ROS_DEBUG_STREAM(
            "Threshold: " << count(start_time, threshold_time) << std::endl
         << "Final: " << count(threshold_time, final_time) << std::endl);
}

} // namespace iarc7_vision
