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

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros_utils/SafeTransformWrapper.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <geometry_msgs/Vector3Stamped.h>

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

/// Chi^2 loss function, where the distance measurement for each datapoint
/// is the distance from the point to the closest of the four vectors
/// parallel, antiparallel, or perpendicular to theta
///
/// Expects that 0 <= theta < pi/2
/// and each datapoint t satisfies -pi/2 <= t < pi/2
static double theta_loss(std::vector<double> thetas, double theta)
{
    double loss = 0;
    for (double t : thetas) {
        double dist = std::numeric_limits<double>::max();
        for (double theta_inc = -M_PI; theta_inc <= M_PI; theta_inc += M_PI/2) {
            dist = std::min(dist, std::abs(t - (theta + theta_inc)));
        }
        loss += std::pow(dist, 2);
    }
    return loss;
}

namespace iarc7_vision {

GridLineEstimator::GridLineEstimator(
        double fov,
        const LineExtractorSettings& line_extractor_settings,
        const GridEstimatorSettings& grid_estimator_settings,
        const GridLineDebugSettings& debug_settings)
    : fov_(fov),
      line_extractor_settings_(line_extractor_settings),
      grid_estimator_settings_(grid_estimator_settings),
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

    // focal length in pixels
    const double focal_length = getFocalLength(image.size(), fov_);

    // TODO: GET HEIGHT FROM SOMEWHERE ELSE, MAKE THIS level_quad
    geometry_msgs::TransformStamped transform;
    ROS_ASSERT(transform_wrapper_.getTransformAtTime(transform,
                                                     "map",
                                                     "bottom_camera_optical",
                                                     time,
                                                     ros::Duration(1.0)));
    geometry_msgs::TransformStamped q_lq_tf;
    //ROS_ASSERT(transform_wrapper_.getTransformAtTime(q_lq_tf,
    //                                                 "level_quad",
    //                                                 "quad",
    //                                                 ros::Time::now(),
    //                                                 ros::Duration(1.0)));
    q_lq_tf = transform;

    double height = transform.transform.translation.z;

    //////////////////////////////////////////////////
    // extract lines from image
    //////////////////////////////////////////////////

    std::vector<cv::Vec2f> lines;
    getLines(lines, image, height);
    ROS_WARN("%lu", lines.size());

    //////////////////////////////////////////////////
    // transform lines into gridlines
    //////////////////////////////////////////////////

    // each line that the camera sees defines a plane (call it P_l) in which
    // the actual line resides. We can find the line in the floor plane
    // (call the line l_f and the plane P_f) by intersecting P_f with P_l.
    //
    // TODO: To make this more robust, we should also check that the line on the
    // floor is within the fov of the camera.  This would have the advantage of
    // automatically filtering out some lines which clearly aren't actually
    // on the floor.
    //
    // To do this, we can send rays through the endpoints of the line in the
    // camera frame and make sure that one of them intersects P_f.

    std::vector<Eigen::Vector3d> pl_normals;
    for (const cv::Vec2f& line : lines) {
        // line goes through the point p, perpendicular to the vector p
        float rho = line[0]; // length of p
        float theta = line[1]; // angle from +x to p

        // vector we rotate around to get P_l's normal vector in the optical
        // frame
        Eigen::Vector3d rot_vector (-sin(theta), cos(theta), 0);
        Eigen::Vector3d pl_normal = Eigen::AngleAxisd(
                                         std::atan(rho / focal_length),
                                         rot_vector)
                                  * Eigen::Vector3d(cos(theta), sin(theta), 0);

        // transform pl_normal into the map frame
        geometry_msgs::Vector3Stamped pl_normal_msg;
        pl_normal_msg.vector.x = pl_normal(0);
        pl_normal_msg.vector.y = pl_normal(1);
        pl_normal_msg.vector.z = pl_normal(2);
        tf2::doTransform(pl_normal_msg, pl_normal_msg, transform);

        pl_normal(0) = pl_normal_msg.vector.x;
        pl_normal(1) = pl_normal_msg.vector.y;
        pl_normal(2) = pl_normal_msg.vector.z;

        // the equation for P_l is then pl_normal * v = pl_normal.z * z_cam,
        // where v is a vector in P_l and z_cam is the distance from the ground
        // to the camera.
        //
        // this means the equation for l_f (in the map frame) is
        // pl_normal.x * x + pl_normal.y * y = pl_normal.z * z_cam
        //
        // because z_cam is the same for all lines, each line is specified by
        // pl_normal alone
        pl_normals.push_back(pl_normal);
    }

    //////////////////////////////////////////////////
    // cluster gridlines by angle
    //////////////////////////////////////////////////

    // extract angles from lines (angles in the list are in [0, pi/2))
    std::vector<double> thetas;
    for (const Eigen::Vector3d& pl_normal : pl_normals) {
        double next_theta = std::atan(-pl_normal(0)/pl_normal(1));
        if (next_theta < 0) {
            next_theta += M_PI/2;
        }
        thetas.push_back(next_theta);
    }

    // do a coarse sweep over angles from 0 to pi/2
    double best_coarse_theta = 0;
    double best_coarse_theta_score = theta_loss(thetas, 0);
    for (double theta = grid_estimator_settings_.theta_step;
         theta < M_PI / 2;
         theta += grid_estimator_settings_.theta_step) {
        double theta_score = theta_loss(thetas, theta);
        if (theta_score < best_coarse_theta_score) {
            best_coarse_theta = theta;
            best_coarse_theta_score = theta_score;
        }
    }

    // do an average around the result of the coarse estimate
    double total = 0;
    for (double theta : thetas) {
        if (theta - best_coarse_theta > M_PI/4) {
            total += theta - M_PI/2;
        } else if (theta - best_coarse_theta < -M_PI/4) {
            total += theta + M_PI/2;
        } else {
            total += theta;
        }
    }
    double best_theta = total / thetas.size();

    // wrap the result into [0, pi/2)
    if (best_theta > M_PI/2) {
        best_theta -= M_PI/2;
    }
    if (best_theta < 0) {
        best_theta += M_PI/2;
    }

    // TODO: Remove outliers here and redo the average without them

    // get current orientation estimate
    geometry_msgs::Vector3Stamped quad_forward;
    quad_forward.vector.x = 1;
    tf2::doTransform(quad_forward, quad_forward, q_lq_tf);
    double current_theta = M_PI/2 + std::atan2(quad_forward.vector.y,
                                               quad_forward.vector.x);
    if (!std::isfinite(current_theta)) {
        ROS_ERROR("The quad seems to be vertical, "
                  "so we have no idea what our yaw is (Yay gimbal lock!)");
        // TODO: handle this so the whole thing doesn't crash
    }

    if (current_theta < 0) {
        current_theta += 2*M_PI;
    }

    // pick the orientation estimate that's closer to our current angle estimate
    double best_theta_quad;
    if (best_theta < M_PI/4) {
        best_theta_quad = current_theta + best_theta;
        if (best_theta_quad >= 2*M_PI) {
            best_theta_quad -= 2*M_PI;
        }
    } else {
        best_theta_quad = current_theta + best_theta - M_PI/2;
        if (best_theta_quad < 0) {
            best_theta_quad += 2*M_PI;
        }
    }

    // convert orientation estimate to yaw
    double yaw = -best_theta_quad;
    if (yaw < 0) {
        yaw += 2*M_PI;
    }

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
