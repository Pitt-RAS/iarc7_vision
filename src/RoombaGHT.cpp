#include "iarc7_vision/RoombaGHT.hpp"

namespace iarc7_vision
{

void RoombaGHT::setup(float pixels_per_meter,
                      float roomba_plate_width,
                      int ght_levels,
                      int ght_dp,
                      int votes_threshold,
                      int template_canny_threshold)
{
    (void)template_canny_threshold;

    float min_dist = pixels_per_meter * roomba_plate_width;

    templ_ = cv::imread("roomba_template.png", cv::IMREAD_GRAYSCALE);

    ght_ = cv::cuda::createGeneralizedHoughGuil();
    ght_->setMinDist(min_dist);
    ght_->setLevels(ght_levels);
    ght_->setDp(ght_dp);
    ght_->setMaxBufferSize(1000); // maximal size of inner buffers
    ght_->setScaleThresh(votes_threshold);
    ght_->setPosThresh(votes_threshold);
    ght_->setMinAngle(0);
    ght_->setMaxAngle(360);
    ght_->setAngleStep(1);

    cv::cuda::GpuMat gpu_templ(templ_);
    ght_->setTemplate(gpu_templ);
}

void RoombaGHT::detect(const cv::cuda::GpuMat& image,
                       const cv::Rect& boundRect,
                       cv::Point2f& pos,
                       double& angle,
                       int camera_canny_threshold)
{
    (void)camera_canny_threshold;
    // First grab the important area of the image
    cv::cuda::GpuMat image_crop(image, boundRect);

    // TRY HSV
    cv::cuda::GpuMat hsv;
    cv::cuda::cvtColor(image_crop, hsv, CV_BGR2HSV);
    cv::cuda::GpuMat hsv_channels[3];
    cv::cuda::split(hsv, hsv_channels);

    // Run the GHT, there should only be one found
    cv::cuda::GpuMat gpu_position;
    std::vector<cv::Vec4f> position;
    std::vector<cv::Vec3i> votes;
    ght_->detect(hsv_channels[1], gpu_position);
    gpu_position.download(position);
    if(!position.size())
        angle = -1;
    pos.x = boundRect.x + position[0][0];
    pos.y = boundRect.y + position[0][1];
    angle = position[0][3];
}

} // namespace iarc7_vision
