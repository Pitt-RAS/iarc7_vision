#include "iarc7_vision/RoombaBlob.hpp"

namespace iarc7_vision
{

// inRange and medianBlur do not exist for the gpu
void RoombaBlob::ThresholdFrame(const cv::Mat& image, cv::Mat& dst){
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);
    cv::Mat hsv_channels[3];
    cv::split(hsv_image, hsv_channels);

    dst = cv::Mat::zeros(image.rows, image.cols, CV_8U);
    cv::Mat range_mask;
    // Green slice (Hue should be 57.43 out of 180)
    cv::inRange(hsv_image, cv::Scalar(47, 20, 15), cv::Scalar(67,255,200), range_mask);
    cv::bitwise_or(dst, range_mask, dst);
    // Upper red slice (Hue should be 3.14 out of 180)
    cv::inRange(hsv_image, cv::Scalar(0, 20, 15), cv::Scalar(8,255,200), range_mask);
    cv::bitwise_or(dst, range_mask, dst);
    // Lower red slice (Hue should be 3.14 out of 180)
    cv::inRange(hsv_image, cv::Scalar(170, 20, 15), cv::Scalar(180,255,200), range_mask);
    cv::bitwise_or(dst, range_mask, dst);
    // cv::inRange(hsv_image, cv::Scalar(47, 20, 15), cv::Scalar(67,255,200), dst);
    // cv::extractChannel(dst3, dst, 0);
    // ROS_ERROR("Number of channels: %d", dst.channels());

    ROS_ASSERT(dst.channels() == 1);
    
    cv::medianBlur(dst, dst, 5);
}

// findContours does not exist for the gpu
void RoombaBlob::BoundMask(const cv::Mat& mask, cv::vector<cv::Rect>& boundRect){
    cv::vector<cv::vector<cv::Point>> contours;
    cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
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

void RoombaBlob::DilateBounds(const cv::Mat& image, cv::vector<cv::Rect>& boundRect){
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

void RoombaBlob::detect(const cv::Mat& image, cv::vector<cv::Rect>& boundRect){
    cv::Mat mask;
    ThresholdFrame(image, mask);
    BoundMask(mask, boundRect);
}

}
