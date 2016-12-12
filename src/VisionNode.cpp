#include <ros/ros.h>

#include "iarc7_vision/GridLineEstimator.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vision");

    ros::NodeHandle nh;

    iarc7_vision::GridLineEstimator gridline_estimator;

    ros::Rate rate (100);
    while (ros::ok() && ros::Time::now() == ros::Time(0)) {
        // wait
        ros::spinOnce();
    }

    gridline_estimator.update();
    while (ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }

    // All is good.
    return 0;
}
