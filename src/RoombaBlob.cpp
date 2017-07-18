#include "iarc7_vision/RoombaBlob.hpp"

namespace iarc7_vision
{

void RoombaBlob::ThresholdFrame(const cv::Mat& image, cv::Mat& dst){
    // std::vector<cv::Mat> bgr;
    // cv::Mat antiblue = image.clone();
    // cv::split(antiblue, bgr);
    // cv::subtract(bgr[2], bgr[0], bgr[2]);
    // cv::subtract(bgr[1], bgr[0], bgr[1]);
    // bgr[0] = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    // cv::merge(bgr, antiblue);
    // ROS_ERROR("There are now %d channels", antiblue.channels());
    // cv::imshow("Anti blue", antiblue);

    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    cv::Mat hsv_channels[3];
    cv::split(hsv_image, hsv_channels);
    cv::Mat dst3;
    // Hue should be 57.43 / 180
    cv::inRange(hsv_image, cv::Scalar(47, 20, 15), cv::Scalar(67,255,200), dst3);
    cv::extractChannel(dst3, dst, 0);
    cv::medianBlur(dst, dst, 5);
}

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
