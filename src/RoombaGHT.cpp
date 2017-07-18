#include "iarc7_vision/RoombaGHT.hpp"

namespace iarc7_vision
{

// Using the Ballard implementation
void RoombaGHT::setup(float pixels_per_meter, float roomba_plate_width, 
                      int ght_levels, int ght_dp, int votes_threshold,
                      int template_canny_threshold){
    float min_dist = pixels_per_meter * roomba_plate_width;
    ght = cv::GeneralizedHough::create(cv::GHT_POSITION | cv::GHT_ROTATION);
    ght->set("minDist", min_dist);
    ght->set("levels", ght_levels);
    ght->set("dp", ght_dp);
    ght->set("votesThreshold", votes_threshold);
    ght->set("minAngle", 0);
    ght->set("maxAngle", 360);
    ght->set("angleStep", 1);

    // Import the template
    templ = imread("roomba_template.png", cv::IMREAD_GRAYSCALE);
    ght->setTemplate(templ, template_canny_threshold);
}

float RoombaGHT::detect(const cv::Mat& image, cv::Rect& boundRect, cv::Point2f& pos, int camera_canny_threshold){

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
  float angle = 0;
  if(position.size() > 0){
    pos.x = boundRect.x + position[0][0];
    pos.y = boundRect.y + position[0][1];
    angle = position[0][3];
  } else {
    return -1;
  }
  return angle;
}

}
