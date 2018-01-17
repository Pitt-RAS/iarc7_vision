#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include "iarc7_vision/cv_utils.hpp"

#include "iarc7_vision/RoombaBlobDetector.hpp"

namespace iarc7_vision
{

RoombaBlobDetector::RoombaBlobDetector(const RoombaEstimatorSettings& settings,
                                       ros::NodeHandle& ph)
    : settings_(settings)
{
    if (settings_.debug_hsv_slice) {
        debug_hsv_slice_pub_ = ph.advertise<sensor_msgs::Image>("hsv_slice", 10);
    }

    if (settings_.debug_contours) {
        debug_contours_pub_ = ph.advertise<sensor_msgs::Image>("contours", 10);
    }

    if (settings_.debug_rects) {
        debug_rects_pub_ = ph.advertise<sensor_msgs::Image>("rects", 10);
    }
}

void RoombaBlobDetector::thresholdFrame(const cv::cuda::GpuMat& image,
                                        cv::cuda::GpuMat& dst)
{
    cv::cuda::GpuMat hsv_image;
    cv::cuda::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);
    cv::cuda::GpuMat hsv_channels[3];
    cv::cuda::split(hsv_image, hsv_channels);

    dst.create(image.rows, image.cols, CV_8U);
    dst.setTo(cv::Scalar(0, 0, 0, 0));

    cv::cuda::GpuMat range_mask;

    // Green slice (Hue should be 57.43 out of 180)
    cv_utils::inRange(hsv_image, cv::Scalar(47, 20, 15), cv::Scalar(67,255,200), range_mask);
    cv::cuda::bitwise_or(dst, range_mask, dst);
    // Upper red slice (Hue should be 3.14 out of 180)
    cv_utils::inRange(hsv_image, cv::Scalar(0, 20, 15), cv::Scalar(8,255,200), range_mask);
    cv::cuda::bitwise_or(dst, range_mask, dst);
    // Lower red slice (Hue should be 3.14 out of 180)
    cv_utils::inRange(hsv_image, cv::Scalar(170, 20, 15), cv::Scalar(180,255,200), range_mask);
    cv::cuda::bitwise_or(dst, range_mask, dst);

    ROS_ASSERT(dst.channels() == 1);
}

// findContours does not exist for the gpu
void RoombaBlobDetector::boundMask(const cv::cuda::GpuMat& mask,
                                   std::vector<cv::Rect>& boundRect)
{
    cv::Mat mask_cpu;
    mask.download(mask_cpu);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_cpu, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

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

        debug_contours_pub_.publish(cv_image);
    }

    boundRect.resize(0); // Clear the vector
    cv::Rect rect;
    for(unsigned int i=0;i<contours.size();i++){
        rect = cv::boundingRect(contours[i]);
        if(rect.area() < 2000)
            continue;
        if(rect.area() > 15000)
            continue;
        if(rect.height > rect.width * 4 || rect.width > rect.height * 4)
            continue;
        boundRect.push_back(rect);
    }
}

void RoombaBlobDetector::dilateBounds(const cv::cuda::GpuMat& image,
                                      std::vector<cv::Rect>& boundRect)
{
    for(unsigned int i=0;i<boundRect.size();i++){
        boundRect[i].x -= 20;
        boundRect[i].y -= 20;
        boundRect[i].width += 40;
        boundRect[i].height += 40;
        if(boundRect[i].x < 0)
            boundRect[i].x = 0;
        if(boundRect[i].y < 0)
            boundRect[i].y = 0;
        if(boundRect[i].x + boundRect[i].width > image.cols)
            boundRect[i].width = image.cols - boundRect[i].x;
        if(boundRect[i].y + boundRect[i].height > image.rows)
            boundRect[i].height = image.rows - boundRect[i].y;
    }

    if (settings_.debug_rects) {
        cv::Mat rect_image = cv::Mat::zeros(image.rows, image.cols, CV_8U);

        for (const auto& rect : boundRect) {
            cv_utils::drawRect(rect_image, rect, cv::Scalar(255));
        }

        const cv_bridge::CvImage cv_image {
            std_msgs::Header(),
            sensor_msgs::image_encodings::MONO8,
            rect_image
        };

        debug_rects_pub_.publish(cv_image);
    }
}

void RoombaBlobDetector::detect(const cv::cuda::GpuMat& image,
                                std::vector<cv::Rect>& boundRect)
{
    cv::cuda::GpuMat mask;
    thresholdFrame(image, mask);

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

    boundMask(mask, boundRect);
    dilateBounds(image, boundRect);
}

} // namespace iarc7_vision
