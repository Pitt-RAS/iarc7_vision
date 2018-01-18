#include "iarc7_vision/RoombaGHT.hpp"

namespace iarc7_vision
{

RoombaGHT::RoombaGHT(const RoombaEstimatorSettings& settings)
    : ght_(cv::cuda::createGeneralizedHoughGuil()),
      settings_(settings)
{
    float min_dist = settings_.pixels_per_meter * settings_.roomba_plate_width;
    ght_->setMinDist(min_dist);

    cv::Mat templ = cv::imread("roomba_template.png", cv::IMREAD_GRAYSCALE);
    cv::cuda::GpuMat gpu_templ(templ);
    ght_->setTemplate(gpu_templ);

    ght_->setMaxBufferSize(1000); // maximal size of inner buffers
    ght_->setMinAngle(0);
    ght_->setMaxAngle(360);

    onSettingsChanged();
}

bool RoombaGHT::detect(const cv::cuda::GpuMat& image,
                       const cv::Rect& boundRect,
                       cv::Point2f& pos,
                       double& angle)
{
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

    if (!position.size()) return false;

    pos.x = boundRect.x + position[0][0];
    pos.y = boundRect.y + position[0][1];
    angle = position[0][3];
    return true;
}

void RoombaGHT::onSettingsChanged()
{
    ght_->setPosThresh(settings_.ght_pos_thresh);
    ght_->setAngleThresh(settings_.ght_angle_thresh);
    ght_->setScaleThresh(settings_.ght_scale_thresh);
    ght_->setCannyLowThresh(settings_.ght_canny_low_thresh);
    ght_->setCannyHighThresh(settings_.ght_canny_high_thresh);
    ght_->setDp(settings_.ght_dp);
    ght_->setLevels(settings_.ght_levels);
    ght_->setAngleStep(settings_.ght_angle_step);
    ght_->setScaleStep(settings_.ght_scale_step);
}

} // namespace iarc7_vision
