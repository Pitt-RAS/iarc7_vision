#include "iarc7_vision/RoombaGHT.hpp"

namespace iarc7_vision
{

// Using the Ballard implementation
void RoombaGHT::setup(float pixels_per_meter, float roomba_plate_width, 
                      int ght_levels, int ght_dp, int votes_threshold,
                      int template_canny_threshold){
    float min_dist = pixels_per_meter * roomba_plate_width;

    templ = imread("roomba_template.png", cv::IMREAD_GRAYSCALE);

    ght = cv::GeneralizedHough::create(cv::GHT_POSITION | cv::GHT_ROTATION);
    ght->set("minDist", min_dist);
    ght->set("levels", ght_levels);
    ght->set("dp", ght_dp);
    ght->set("votesThreshold", votes_threshold);
    ght->set("minAngle", 0);
    ght->set("maxAngle", 360);
    ght->set("angleStep", 1);
    ght->setTemplate(templ, template_canny_threshold);

    useGpu = cv::gpu::getCudaEnabledDeviceCount();
    if(useGpu){
        gpu_ght = cv::gpu::GeneralizedHough_GPU::create(cv::GHT_POSITION | cv::GHT_ROTATION);
        gpu_ght->set("minDist", min_dist);
        gpu_ght->set("levels", ght_levels);
        gpu_ght->set("dp", ght_dp);
        gpu_ght->set("maxSize", 1000); // maximal size of inner buffers
        gpu_ght->set("votesThreshold", votes_threshold);
        gpu_ght->set("minAngle", 0);
        gpu_ght->set("maxAngle", 360);
        gpu_ght->set("angleStep", 1);
        cv::gpu::GpuMat gpu_templ(templ);
        gpu_ght->setTemplate(gpu_templ, template_canny_threshold);
    }
}

// You can use this whether or not GPU is available
float RoombaGHT::detect(const cv::Mat& image, cv::Rect& boundRect,
                        cv::Point2f& pos, int camera_canny_threshold){
    // Use the GPU if available
    if(useGpu){
        cv::gpu::GpuMat gpu_image(image);
        return detect(gpu_image, boundRect, pos, camera_canny_threshold);
    }

    // First grab the important area of the image
    cv::Mat image_crop = image(boundRect);

    // TRY HSV
    cv::Mat hsv;
    cv::cvtColor(image_crop, hsv, CV_BGR2HSV);
    cv::Mat hsv_channels[3];
    cv::split(hsv, hsv_channels);

    // Run the GHT, there should only be one found
    cv::vector<cv::Vec4f> position;
    cv::vector<cv::Vec3i> votes;
    ght->detect(hsv_channels[1], position, votes, camera_canny_threshold);
    if(!position.size())
        return -1;
    pos.x = boundRect.x + position[0][0];
    pos.y = boundRect.y + position[0][1];
    return position[0][3];
}

float RoombaGHT::detect(const cv::gpu::GpuMat& image, cv::Rect& boundRect,
                        cv::Point2f& pos, int camera_canny_threshold){
    ROS_ASSERT(useGpu);

    // First grab the important area of the image
    cv::gpu::GpuMat image_crop(image, boundRect);

    // TRY HSV
    cv::gpu::GpuMat hsv;
    cv::gpu::cvtColor(image_crop, hsv, CV_BGR2HSV);
    cv::gpu::GpuMat hsv_channels[3];
    cv::gpu::split(hsv, hsv_channels);

    // Run the GHT, there should only be one found
    cv::gpu::GpuMat gpu_position;
    cv::vector<cv::Vec4f> position;
    cv::vector<cv::Vec3i> votes;
    gpu_ght->detect(hsv_channels[1], gpu_position, camera_canny_threshold);
    gpu_ght->download(gpu_position, position);
    if(!position.size())
        return -1;
    pos.x = boundRect.x + position[0][0];
    pos.y = boundRect.y + position[0][1];
    return position[0][3];
}

}
