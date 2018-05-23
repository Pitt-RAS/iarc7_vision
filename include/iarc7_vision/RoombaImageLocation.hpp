#ifndef ROOMBA_IMAGE_LOCATION_HPP_
#define ROOMBA_IMAGE_LOCATION_HPP_

#include <ros/ros.h>

namespace iarc7_vision {

struct RoombaImageLocation
{
    // Range from 0-1 and indicate the distance relative to the image
    // width assuming square pixels
    double x;
    double y;
    double radius;

    bool point_on_roomba(double p_x, double p_y, int image_width) const {
        double s_x = p_x / static_cast<double>(image_width);
        double s_y = p_y / static_cast<double>(image_width);
        double s_r = std::sqrt(std::pow(s_x - x, 2) + std::pow(s_y - y, 2));

        return s_r <= radius;
    }
};

}

#endif
