#include "iarc7_vision/GridLineEstimator.hpp"

// BAD HEADERS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Geometry>
#pragma GCC diagnostic pop
// END BAD HEADERS

#include <cmath>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros_utils/SafeTransformWrapper.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

static void drawLines(const std::vector<cv::Vec2f>& lines, cv::Mat image)
{
    for (auto& line : lines) {
        float rho = line[0], theta = line[1];
        cv::Point pt1, pt2;
        double a = cos(theta);
        double b = sin(theta);
        double x0 = image.size().width/2 + a*rho;
        double y0 = image.size().height/2 + b*rho;
        double line_dist = 4*std::max(image.size().width, image.size().height);
        pt1.x = cvRound(x0 + line_dist*(-b));
        pt1.y = cvRound(y0 + line_dist*(a));
        pt2.x = cvRound(x0 - line_dist*(-b));
        pt2.y = cvRound(y0 - line_dist*(a));
        cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 3, CV_AA);
    }
}

namespace iarc7_vision {

GridLineEstimator::GridLineEstimator(
        double fov,
        const LineExtractorSettings& line_extractor_settings,
        const GridLineDebugSettings& debug_settings)
    : fov_(fov),
      line_extractor_settings_(line_extractor_settings),
      debug_settings_(debug_settings),
      transform_wrapper_()
{
    ros::NodeHandle local_nh ("grid_line_estimator");

    if (debug_settings_.debug_edges) {
        debug_edges_pub_ = local_nh.advertise<sensor_msgs::Image>("edges", 10);
    }

    if (debug_settings_.debug_lines) {
        debug_lines_pub_ = local_nh.advertise<sensor_msgs::Image>("lines", 10);
    }
}

void GridLineEstimator::update(const cv::Mat& image, const ros::Time& time)
{
    // calculate necessary parameters

    // TODO: GET HEIGHT FROM SOMEWHERE ELSE, MAKE THIS level_quad
    geometry_msgs::TransformStamped transform;
    ROS_ASSERT(transform_wrapper_.getTransformAtTime(transform,
                                                     "map",
                                                     "bottom_camera_optical",
                                                     time,
                                                     ros::Duration(1.0)));
    double height = transform.transform.translation.z;

    //////////////////////////////////////////////////
    // extract lines
    //////////////////////////////////////////////////
    std::vector<cv::Vec2f> lines;
    getLines(lines, image, height);
    ROS_WARN("%lu", lines.size());

    //////////////////////////////////////////////////
    // compute tranformation from lines to gridlines
    //////////////////////////////////////////////////
    /*
    f = transform (1 0 0) from camera frame to level_quad
    */

    //////////////////////////////////////////////////
    // transform lines into gridlines
    //////////////////////////////////////////////////
    /*
    for each line:
        v = (0 sin(theta) cos(theta))
        global_v = transform v from camera frame to level_quad
        offset = (0 -r*cos(theta) r*sin(theta))
        global_offset = transform offset from camera_frame to level_quad
        ray1 = f + global_offset
        ray2 = f + global_offset + line_time * global_v

        // think about what happens when these don't intersect
        p1 = intersection of ray1 with the plane z=0
        p2 = intersection of ray2 with the plane z=0
        result = linebetween p1 and p2
    */

    //////////////////////////////////////////////////
    // cluster gridlines
    //////////////////////////////////////////////////
    /*
    sort gridlines by angle
    run k-means with k=2? or run sliding window over list and look for high counts?
    maybe repeatedly pair each line with the nearest line or nearest cluster?

    make a group for each direction and throw out outliers (lines more than a constant number of standard deviations away)
    */

    //////////////////////////////////////////////////
    // return estimate
    //////////////////////////////////////////////////
}

double GridLineEstimator::getFocalLength(const cv::Size& img_size, double fov)
{
    return std::hypot(img_size.width/2.0, img_size.height/2.0)
         / std::tan(fov / 2.0);
}

void GridLineEstimator::getLines(std::vector<cv::Vec2f>& lines,
                                 const cv::Mat& image,
                                 double height)
{
    // m/px = camera_height / focal_length;
    double meters_per_px = height / getFocalLength(image.size(), fov_);

    // desired m_px used to keep kernel sizes relative to our features
    double desired_meters_per_px = 1.0/250.0;
    double scale_factor = meters_per_px / desired_meters_per_px;

    if (cv::gpu::getCudaEnabledDeviceCount() == 0) {
        ROS_WARN_ONCE("Doing OpenCV operations on CPU");

        cv::resize(image, image_sized_, cv::Size(), scale_factor, scale_factor);
        cv::cvtColor(image_sized_, image_hsv_, CV_BGR2HSV);
        cv::split(image_hsv_, image_hsv_channels_);

        cv::Canny(image_hsv_channels_[2],
                  image_edges_,
                  line_extractor_settings_.canny_low_threshold,
                  line_extractor_settings_.canny_high_threshold,
                  line_extractor_settings_.canny_sobel_size);

        double hough_threshold = image_edges_.size().height
                               * line_extractor_settings_.hough_thresh_fraction;
        cv::HoughLines(image_edges_,
                       lines,
                       line_extractor_settings_.hough_rho_resolution,
                       line_extractor_settings_.hough_theta_resolution,
                       hough_threshold);

        // rescale lines back to original image size
        for (cv::Vec2f& line : lines) {
            line[0] *= static_cast<float>(image.size().height)
                     / image_edges_.size().height;
        }
    } else {
        gpu_image_.upload(image);

        cv::gpu::resize(gpu_image_,
                        gpu_image_sized_,
                        cv::Size(),
                        scale_factor,
                        scale_factor);
        cv::gpu::cvtColor(gpu_image_sized_, gpu_image_hsv_, CV_BGR2HSV);
        cv::gpu::split(gpu_image_hsv_, gpu_image_hsv_channels_);

        cv::gpu::Canny(gpu_image_hsv_channels_[2],
                       gpu_image_edges_,
                       line_extractor_settings_.canny_low_threshold,
                       line_extractor_settings_.canny_high_threshold,
                       line_extractor_settings_.canny_sobel_size);

        double hough_threshold = gpu_image_edges_.size().height
                               * line_extractor_settings_.hough_thresh_fraction;
        cv::gpu::HoughLines(gpu_image_edges_,
                            gpu_lines_,
                            gpu_hough_buf_,
                            line_extractor_settings_.hough_rho_resolution,
                            line_extractor_settings_.hough_theta_resolution,
                            hough_threshold);

        cv::gpu::HoughLinesDownload(gpu_lines_, lines);

        // rescale lines back to original image size
        for (cv::Vec2f& line : lines) {
            line[0] *= static_cast<float>(image.size().height)
                     / gpu_image_edges_.size().height;
        }

        if (debug_settings_.debug_edges) {
            gpu_image_edges_.download(image_edges_);
        }
    }

    // shift rho to measure distance from center instead of distance from
    // top left corner
    Eigen::Vector2d corner_to_center (image.size().width/2, image.size().height/2);
    for (cv::Vec2f& line : lines) {
        Eigen::Vector2d normal_dir (std::cos(line[1]), std::sin(line[1]));
        line[0] -= corner_to_center.dot(normal_dir);
    }

    if (debug_settings_.debug_edges) {
        cv_bridge::CvImage cv_image {
            std_msgs::Header(),
            sensor_msgs::image_encodings::MONO8,
            image_edges_
        };

        debug_edges_pub_.publish(cv_image.toImageMsg());
    }

    if (debug_settings_.debug_lines) {
        cv::Mat image_lines = image.clone();
        drawLines(lines, image_lines);
        cv_bridge::CvImage cv_image {
            std_msgs::Header(),
            sensor_msgs::image_encodings::RGBA8,
            image_lines
        };
        debug_lines_pub_.publish(cv_image.toImageMsg());
    }
}

} // namespace iarc7_vision
