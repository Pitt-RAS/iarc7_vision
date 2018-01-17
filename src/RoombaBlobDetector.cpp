#include "iarc7_vision/RoombaBlobDetector.hpp"

#include "iarc7_vision/cv_utils.hpp"

namespace iarc7_vision
{

RoombaBlobDetector::RoombaBlobDetector(const RoombaEstimatorSettings& settings)
    : settings_(settings)
{
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
}

void RoombaBlobDetector::detect(const cv::cuda::GpuMat& image,
                                std::vector<cv::Rect>& boundRect)
{
    cv::cuda::GpuMat mask;
    thresholdFrame(image, mask);
    boundMask(mask, boundRect);
    dilateBounds(image, boundRect);
}

}
