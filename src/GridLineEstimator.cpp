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
#include <boost/format.hpp>
#include <cmath>
#include <exception>
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

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <iarc7_msgs/Float64Stamped.h>
#include <visualization_msgs/Marker.h>

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
static double theta_loss(const std::vector<double>& thetas, double theta)
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
        const LineExtractorSettings& line_extractor_settings,
        const GridEstimatorSettings& grid_estimator_settings,
        const GridLineDebugSettings& debug_settings)
    : line_extractor_settings_(line_extractor_settings),
      grid_estimator_settings_(grid_estimator_settings),
      debug_settings_(debug_settings),
      transform_wrapper_()
{
    ros::NodeHandle local_nh ("grid_line_estimator");

    if (debug_settings_.debug_direction) {
        debug_direction_marker_pub_
            = local_nh.advertise<visualization_msgs::Marker>("direction", 1);
    }

    if (debug_settings_.debug_edges) {
        debug_edges_pub_ = local_nh.advertise<sensor_msgs::Image>("edges", 10);
    }

    if (debug_settings_.debug_lines) {
        debug_lines_pub_ = local_nh.advertise<sensor_msgs::Image>("lines", 10);
    }

    if (debug_settings_.debug_line_markers) {
        debug_line_markers_pub_ = local_nh.advertise<visualization_msgs::Marker>(
                "line_markers", 1);
    }

    pose_pub_
        = local_nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose",
                                                                       10);

    yaw_pub_ = local_nh.advertise<iarc7_msgs::Float64Stamped>("line_yaw", 10);
}

void GridLineEstimator::update(const cv::Mat& image, const ros::Time& time)
{
    ROS_DEBUG("last filtered position %f", last_filtered_position_(2));

    if (last_filtered_position_stamp_ == ros::Time(0)
     || (time - last_filtered_position_stamp_).toSec()
         > grid_estimator_settings_.allowed_position_stamp_error) {
        updateFilteredPosition(time);
    }

    if (last_filtered_position_stamp_ == ros::Time(0)
     || (time - last_filtered_position_stamp_).toSec()
         > grid_estimator_settings_.allowed_position_stamp_error) {
            ROS_ERROR_STREAM(
                "Skipping frame because we don't have a transform at time "
             << time);
            return;
    }

    if (last_filtered_position_(2)
            >= grid_estimator_settings_.min_extraction_altitude) {
        try {
            processImage(image, time);
        } catch (const std::exception& ex) {
            ROS_ERROR_STREAM("Caught exception processing image: "
                          << ex.what());
        }
    } else {
        ROS_WARN("Height (%f) is below min processing height (%f)",
                 last_filtered_position_(2),
                 grid_estimator_settings_.min_extraction_altitude);
    }

    updateFilteredPosition(time);
}

double GridLineEstimator::getCurrentTheta(const ros::Time& time) const
{
    geometry_msgs::TransformStamped q_lq_tf;
    if (!transform_wrapper_.getTransformAtTime(q_lq_tf,
                                               "level_quad",
                                               "quad",
                                               time,
                                               ros::Duration(1.0))) {
        throw ros::Exception("Failed to fetch transform");
    }

    geometry_msgs::Vector3Stamped quad_forward;
    quad_forward.vector.x = 1;
    tf2::doTransform(quad_forward, quad_forward, q_lq_tf);
    double current_theta = M_PI/2 - std::atan2(quad_forward.vector.y,
                                               quad_forward.vector.x);
    if (!std::isfinite(current_theta)) {
        ROS_ERROR("The quad seems to be vertical, "
                  "so we have no idea what our yaw is (Yay gimbal lock!)");
        // TODO: handle this so the whole thing doesn't crash
    }

    if (current_theta >= 2*M_PI) {
        current_theta -= 2*M_PI;
    }

    return current_theta;
}

double GridLineEstimator::getFocalLength(const cv::Size& img_size, double fov)
{
    return std::hypot(img_size.width/2.0, img_size.height/2.0)
         / std::tan(fov / 2.0);
}

void GridLineEstimator::getLines(std::vector<cv::Vec2f>& lines,
                                 const cv::Mat& image,
                                 double height) const
{
    // m/px = camera_height / focal_length;
    double current_meters_per_px = height
                         / getFocalLength(image.size(),
                                          line_extractor_settings_.fov);

    // desired m_px used to keep kernel sizes relative to our features
    double desired_meters_per_px = 1.0
                                 / line_extractor_settings_.pixels_per_meter;

    double scale_factor = current_meters_per_px / desired_meters_per_px;

    cv::Mat image_edges;
    if (cv::gpu::getCudaEnabledDeviceCount() == 0) {
        ROS_WARN_ONCE("Doing OpenCV operations on CPU");

        cv::Mat image_sized;
        cv::Mat image_hsv;
        cv::Mat image_hsv_channels[3];
        cv::resize(image, image_sized, cv::Size(), scale_factor, scale_factor);
        cv::cvtColor(image_sized, image_hsv, CV_RGB2HSV);
        cv::split(image_hsv, image_hsv_channels);

        cv::Canny(image_hsv_channels[2],
                  image_edges,
                  line_extractor_settings_.canny_low_threshold,
                  line_extractor_settings_.canny_high_threshold,
                  line_extractor_settings_.canny_sobel_size);

        double hough_threshold = image_edges.size().height
                               * line_extractor_settings_.hough_thresh_fraction;
        cv::HoughLines(image_edges,
                       lines,
                       line_extractor_settings_.hough_rho_resolution,
                       line_extractor_settings_.hough_theta_resolution,
                       hough_threshold);

        // rescale lines back to original image size
        for (cv::Vec2f& line : lines) {
            line[0] *= static_cast<float>(image.size().height)
                     / image_edges.size().height;
        }
    } else {
        cv::gpu::HoughLinesBuf gpu_hough_buf;
        cv::gpu::GpuMat gpu_image;
        cv::gpu::GpuMat gpu_image_sized;
        cv::gpu::GpuMat gpu_image_hsv;
        cv::gpu::GpuMat gpu_image_edges;
        cv::gpu::GpuMat gpu_lines;
        cv::gpu::GpuMat gpu_image_hsv_channels[3];

        gpu_image.upload(image);
        ROS_DEBUG("Scale factor %f", scale_factor);

        cv::gpu::resize(gpu_image,
                        gpu_image_sized,
                        cv::Size(),
                        scale_factor,
                        scale_factor);
        cv::gpu::cvtColor(gpu_image_sized, gpu_image_hsv, CV_RGB2HSV);

        cv::gpu::split(gpu_image_hsv, gpu_image_hsv_channels);

        // Turns out that cpu canny isn't the same as gpu Canny
        // Using a seperate sobel operator with a scaling factor
        // achieves similar resuts to the CPU canny
        cv::gpu::GpuMat dx;
        cv::gpu::GpuMat dy;
        cv::gpu::Sobel(gpu_image_hsv_channels[2], dx, CV_32S, 1, 0, line_extractor_settings_.canny_sobel_size, 0.5, cv::BORDER_REPLICATE);
        cv::gpu::Sobel(gpu_image_hsv_channels[2], dy, CV_32S, 0, 1, line_extractor_settings_.canny_sobel_size, 0.5, cv::BORDER_REPLICATE);

        cv::gpu::Canny(dx,
                       dy,
                       gpu_image_edges,
                       line_extractor_settings_.canny_low_threshold,
                       line_extractor_settings_.canny_high_threshold);

        // Commented out integrated gpu canny with sobel
        /*cv::gpu::Canny(gpu_image_hsv_channels[2],
                       gpu_image_edges,
                       line_extractor_settings_.canny_low_threshold,
                       line_extractor_settings_.canny_high_threshold,
                       line_extractor_settings_.canny_sobel_size, true);*/

        // Commented out cpu only canny
        /*cv::Canny((cv::Mat)gpu_image_hsv_channels[2],
                  image_edges,
                  line_extractor_settings_.canny_low_threshold,
                  line_extractor_settings_.canny_high_threshold,
                  line_extractor_settings_.canny_sobel_size);

        gpu_image_edges.upload(image_edges);*/

        double hough_threshold = gpu_image_edges.size().height
                               * line_extractor_settings_.hough_thresh_fraction;
        cv::gpu::HoughLines(gpu_image_edges,
                            gpu_lines,
                            gpu_hough_buf,
                            line_extractor_settings_.hough_rho_resolution,
                            line_extractor_settings_.hough_theta_resolution,
                            hough_threshold);

        cv::gpu::HoughLinesDownload(gpu_lines, lines);

        // rescale lines back to original image size
        for (cv::Vec2f& line : lines) {
            line[0] *= static_cast<float>(image.size().height)
                     / gpu_image_edges.size().height;
        }

        if (debug_settings_.debug_edges) {
            gpu_image_edges.download(image_edges);
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
            image_edges
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

void GridLineEstimator::getPlanesForImageLines(
        const std::vector<cv::Vec2f>& image_lines,
        const ros::Time& time,
        double focal_length,
        std::vector<Eigen::Vector3d>& pl_normals) const
{
    geometry_msgs::TransformStamped camera_to_lq_transform;
    if (!transform_wrapper_.getTransformAtTime(camera_to_lq_transform,
                                               "level_quad",
                                               "bottom_camera_optical",
                                               time,
                                               ros::Duration(1.0))) {
        throw ros::Exception("Failed to fetch transform");
    }

    // TODO: To make this more robust, we should also check that the line on the
    // floor is within the fov of the camera.  This would have the advantage of
    // automatically filtering out some lines which clearly aren't actually
    // on the floor.
    //
    // To do this, we can send rays through the endpoints of the line in the
    // camera frame and make sure that one of them intersects P_f.

    pl_normals.clear();
    for (const cv::Vec2f& line : image_lines) {
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

        // transform pl_normal into the map-oriented frame
        geometry_msgs::Vector3Stamped pl_normal_msg;
        pl_normal_msg.vector.x = pl_normal(0);
        pl_normal_msg.vector.y = pl_normal(1);
        pl_normal_msg.vector.z = pl_normal(2);

        double pl_normal_len = pl_normal.norm();

        tf2::doTransform(pl_normal_msg, pl_normal_msg, camera_to_lq_transform);

        pl_normal(0) = pl_normal_msg.vector.x;
        pl_normal(1) = pl_normal_msg.vector.y;
        pl_normal(2) = pl_normal_msg.vector.z;

        if (!(std::abs(pl_normal_len - pl_normal.norm()) < 0.0001)) {
            throw ros::Exception(str(boost::format("%f %f")
                               % pl_normal_len % pl_normal.norm()));
        }

        pl_normals.push_back(pl_normal);
    }
}

double GridLineEstimator::getThetaForPlanes(
        const std::vector<Eigen::Vector3d>& pl_normals) const
{
    if (pl_normals.size() == 0) {
        throw ros::Exception("No planes to process");
    }

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
    if (best_theta >= M_PI/2) {
        best_theta -= M_PI/2;
    }
    if (best_theta < 0) {
        best_theta += M_PI/2;
    }

    if (best_theta < 0 || best_theta >= M_PI/2) {
        throw ros::Exception(str(boost::format("best_theta: %f") % best_theta));
    }

    // TODO: Remove outliers here and redo the average without them

    return best_theta;
}

void GridLineEstimator::getUnAltifiedDistancesFromLines(
        double theta,
        const std::vector<Eigen::Vector3d>& para_line_normals,
        const std::vector<Eigen::Vector3d>& perp_line_normals,
        std::vector<double>& para_signed_dists,
        std::vector<double>& perp_signed_dists)
{
    para_signed_dists.clear();
    perp_signed_dists.clear();

    Eigen::Vector2d theta_vect {
        std::cos(theta),
        std::sin(theta)
    };

    Eigen::Vector2d theta_vect_perp {
        std::cos(theta + M_PI/2),
        std::sin(theta + M_PI/2)
    };

    for (const Eigen::Vector3d& pl_normal : para_line_normals) {
        para_signed_dists.push_back(pl_normal(2)
                                  / theta_vect_perp.dot(pl_normal.head<2>()));
    }

    for (const Eigen::Vector3d& pl_normal : perp_line_normals) {
        perp_signed_dists.push_back(pl_normal(2)
                                  / theta_vect.dot(pl_normal.head<2>()));
    }
}

void GridLineEstimator::get1dGridShift(
        const std::vector<double>& wrapped_dists,
        double& value,
        double& variance) const
{
    if (wrapped_dists.size() == 0) {
        throw ros::Exception("No dists to process");
    }

    double grid_spacing = grid_estimator_settings_.grid_spacing;
    double line_thickness = grid_estimator_settings_.grid_line_thickness;

    // Take coarse samples and use min cost one
    double min_cost = std::numeric_limits<double>::max();
    double best_guess = -1;
    for (double sample = 0;
         sample < grid_spacing;
         sample += grid_estimator_settings_.grid_step) {
        double cost = gridLoss(wrapped_dists, sample);
        if (cost < min_cost) {
            min_cost = cost;
            best_guess = sample;
        }
    }
    if (best_guess < 0 || best_guess >= grid_spacing) {
        throw ros::Exception("best_guess out of bounds");
    }

    // TODO: detect case where we only have one side of one line, and therefore
    // can't tell which side of that line we have

    for (int i = 0;
         i < grid_estimator_settings_.grid_translation_mean_iterations;
         i++) {

        double total = 0;
        for (double sample_dist : wrapped_dists) {
            double min_dist = std::numeric_limits<double>::max();
            double total_inc;
            for (double line_side : {-1, 1}) {
                for (int cell_inc : {-1, 0, 1}) {
                    double test_sample = sample_dist
                                       + line_side * line_thickness/2
                                       + cell_inc * grid_spacing;
                    double dist = std::abs(test_sample - best_guess);
                    if (dist < min_dist) {
                        min_dist = dist;
                        total_inc = test_sample;
                    }
                }
            }

            if (!std::isfinite(total_inc)) {
                throw ros::Exception(str(boost::format("total_inc is %f")
                                   % total_inc));
            }

            total += total_inc;
        }

        best_guess = total / wrapped_dists.size();
        if (best_guess >= grid_spacing) {
            best_guess -= grid_spacing;
        }
        if (best_guess < 0) {
            best_guess += grid_spacing;
        }
    }

    value = best_guess;
    if (wrapped_dists.size() > 1) {
        variance = gridLoss(wrapped_dists, value)
                 / (wrapped_dists.size() * (wrapped_dists.size()-1));
    } else {
        variance = std::pow(line_thickness, 2);
    }
}

void GridLineEstimator::get2dPosition(
        const std::vector<double>& para_signed_dists,
        const std::vector<double>& perp_signed_dists,
        double theta,
        double height_estimate,
        const Eigen::Vector2d& position_estimate,
        Eigen::Vector2d& position,
        Eigen::Matrix2d& covariance) const
{
    double grid_spacing = grid_estimator_settings_.grid_spacing;
    if (theta < 0 || theta >= 2*M_PI) {
        throw ros::Exception(str(boost::format("theta (%f) out of bounds")
                           % theta));
    }

    // Wrap all distances into [0, grid_spacing)
    std::vector<double> para_wrapped_dists;
    for (double dist : para_signed_dists) {
        dist = std::fmod(dist * height_estimate, grid_spacing);
        para_wrapped_dists.push_back(dist);
        if (para_wrapped_dists.back() < 0) {
            para_wrapped_dists.back() += grid_spacing;
        }
    }

    std::vector<double> perp_wrapped_dists;
    for (double dist : perp_signed_dists) {
        dist = std::fmod(dist * height_estimate, grid_spacing);
        perp_wrapped_dists.push_back(dist);
        if (perp_wrapped_dists.back() < 0) {
            perp_wrapped_dists.back() += grid_spacing;
        }
    }

    for (double dist : para_wrapped_dists) {
        if (dist < 0 || dist >= grid_spacing) {
            throw ros::Exception("dist is out of bounds");
        }
    }

    for (double dist : perp_wrapped_dists) {
        if (dist < 0 || dist >= grid_spacing) {
            throw ros::Exception("dist is out of bounds");
        }
    }

    std::ostringstream para_stream;
    for (double d : para_wrapped_dists) para_stream << d << " ";
    ROS_DEBUG_STREAM("GridLineEstimator parallel wrapped distances: "
                  << para_stream.str());

    std::ostringstream perp_stream;
    for (double d : perp_wrapped_dists) perp_stream << d << " ";
    ROS_DEBUG_STREAM("GridLineEstimator perpendicular wrapped distances: "
                  << perp_stream.str());

    // Get estimates and variances for grid translation in
    // parallel/perpendicular directions
    Eigen::Vector2d shift_estimate;
    Eigen::Matrix2d shift_estimate_covariance;
    bool use_last_perp_shift = false;
    bool use_last_para_shift = false;
    if (perp_wrapped_dists.size() != 0) {
        get1dGridShift(perp_wrapped_dists,
                       shift_estimate(0),
                       shift_estimate_covariance(0, 0));
    } else {
        use_last_perp_shift = true;
        shift_estimate_covariance(0, 0) = grid_spacing;
    }

    if (para_wrapped_dists.size() != 0) {
        get1dGridShift(para_wrapped_dists,
                       shift_estimate(1),
                       shift_estimate_covariance(1, 1));
    } else {
        use_last_para_shift = true;
        shift_estimate_covariance(1, 1) = grid_spacing;
    }

    if (shift_estimate(0) < 0 || shift_estimate(0) >= grid_spacing) {
        throw ros::Exception("shift_estimate out of bounds");
    }

    if (shift_estimate(1) < 0 || shift_estimate(1) >= grid_spacing) {
        throw ros::Exception("shift_estimate out of bounds");
    }

    // Convert estimate of grid position into estimate of quad position
    Eigen::Vector2d wrapped_quad_position
        = grid_estimator_settings_.grid_zero_offset - shift_estimate;
    for (size_t i = 0; i < 2; i++) {
        wrapped_quad_position(i) = std::fmod(wrapped_quad_position(i),
                                             grid_spacing);
        if (wrapped_quad_position(i) < 0) {
            wrapped_quad_position(i) += grid_spacing;
        }
    }

    if (wrapped_quad_position(0) < 0
     || wrapped_quad_position(0) >= grid_spacing) {
        throw ros::Exception("wrapped_quad_position out of bounds");
    }

    if (wrapped_quad_position(1) < 0
     || wrapped_quad_position(1) >= grid_spacing) {
        throw ros::Exception("wrapped_quad_position out of bounds");
    }

    // Find new position estimate closest to last position
    Eigen::Vector2d wrapped_position_estimate;
    for (size_t i = 0; i < 2; i++) {
        wrapped_position_estimate(i) = std::fmod(position_estimate(i),
                                                 grid_spacing);
        if (wrapped_position_estimate(i) < 0) {
            wrapped_position_estimate(i) += grid_spacing;
        }

        if (wrapped_position_estimate(i) < 0
         || wrapped_position_estimate(i) >= grid_spacing) {
            throw ros::Exception("wrapped_position_estimate out of bounds");
        }


        int cell_shift;
        if (wrapped_quad_position(i) - wrapped_position_estimate(i)
                < -grid_spacing/2) {
            // we've moved into the next cell
            cell_shift = 1;
        } else if (wrapped_quad_position(i) - wrapped_position_estimate(i)
                < grid_spacing/2) {
            // we're in the same cell
            cell_shift = 0;
        } else {
            // we've moved into the previous cell
            cell_shift = -1;
        }

        position(i) = (std::floor(position_estimate(i) / grid_spacing) + cell_shift)
                        * grid_spacing
                    + wrapped_quad_position(i);
    }

    if (use_last_perp_shift) {
        position(0) = position_estimate(0);
    }

    if (use_last_para_shift) {
        position(1) = position_estimate(1);
    }

    covariance = shift_estimate_covariance;
}

double GridLineEstimator::gridLoss(const std::vector<double>& wrapped_dists,
                                   double dist) const
{
    double grid_spacing = grid_estimator_settings_.grid_spacing;

    double smaller_dist = dist - grid_estimator_settings_.grid_line_thickness/2;
    if (smaller_dist < 0) {
        smaller_dist += grid_spacing;
    }

    double larger_dist = dist + grid_estimator_settings_.grid_line_thickness/2;
    if (larger_dist >= grid_spacing) {
        larger_dist -= grid_spacing;
    }

    double result = 0;
    for (double sample_dist : wrapped_dists) {
        result += std::min({
                    std::pow(smaller_dist - sample_dist + grid_spacing, 2),
                    std::pow(smaller_dist - sample_dist, 2),
                    std::pow(smaller_dist - sample_dist - grid_spacing, 2),
                    std::pow(larger_dist - sample_dist + grid_spacing, 2),
                    std::pow(larger_dist - sample_dist, 2),
                    std::pow(larger_dist - sample_dist - grid_spacing, 2)
                });
    }
    return result;
}

void GridLineEstimator::processImage(const cv::Mat& image,
                                     const ros::Time& time) const
{
    const double height = last_filtered_position_(2);

    // Extract lines from image
    std::vector<cv::Vec2f> lines;
    getLines(lines, image, height);
    ROS_INFO("Number of lines extracted: %lu", lines.size());

    // Don't process further if we don't have any lines
    if (lines.size() == 0) return;

    // Transform lines into gridlines
    std::vector<Eigen::Vector3d> pl_normals;
    getPlanesForImageLines(lines,
                           time,
                           getFocalLength(image.size(),
                                          line_extractor_settings_.fov),
                           pl_normals);

    // Publish gridlines
    visualization_msgs::Marker lines_marker;
    lines_marker.header.stamp = ros::Time::now();
    lines_marker.header.frame_id = "map";
    lines_marker.ns = "lines_marker_ns";
    lines_marker.id = 0;
    lines_marker.type = visualization_msgs::Marker::LINE_LIST;
    lines_marker.action = visualization_msgs::Marker::MODIFY;
    for (const Eigen::Vector3d& pl_normal : pl_normals) {
        geometry_msgs::Point p;
        p.x = 10;
        p.y = (pl_normal(2)*height - pl_normal(0)*p.x) / pl_normal(1);
        lines_marker.points.push_back(p);
        p.x = -10;
        p.y = (pl_normal(2)*height - pl_normal(0)*p.x) / pl_normal(1);
        lines_marker.points.push_back(p);
    }
    lines_marker.color.a = 1;
    lines_marker.color.r = 0;
    lines_marker.color.g = 1;
    lines_marker.color.b = 1;
    lines_marker.scale.x = 0.02;
    if (debug_settings_.debug_line_markers) {
        debug_line_markers_pub_.publish(lines_marker);
    }


    if (!debug_settings_.debug_line_detector) {
        processLines(height, pl_normals, time);
    }
}

void GridLineEstimator::processLines(
        double height,
        const std::vector<Eigen::Vector3d>& pl_normals,
        const ros::Time& time) const
{
    //////////////////////////////////////////////////
    // cluster gridlines by angle
    //////////////////////////////////////////////////

    // For now, our localization has a better estimate of the grid orientation
    // than the camera does, so we just use that
    const double best_theta = getThetaForPlanes(pl_normals);

    // get current orientation estimate
    const double current_theta = getCurrentTheta(time);

    // Pick the orientation estimate that's closer to our current angle estimate
    double best_theta_quad; // angle in [0, 2pi)
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

    // Convert orientation estimate to yaw
    double yaw = -best_theta_quad;
    if (yaw < 0) {
        yaw += 2*M_PI;
    }

    publishYaw(yaw, time);

    ROS_DEBUG("Orientation error in GridLineEstimator: %f",
              std::min({std::abs(best_theta_quad - current_theta),
                        std::abs(best_theta_quad - current_theta - 2*M_PI),
                        std::abs(best_theta_quad - current_theta + 2*M_PI)}));

    visualization_msgs::Marker direction_marker;
    direction_marker.header.stamp = ros::Time::now();
    direction_marker.header.frame_id = "level_quad";
    direction_marker.ns = "direction_marker_ns";
    direction_marker.id = 0;
    direction_marker.type = visualization_msgs::Marker::ARROW;
    direction_marker.action = visualization_msgs::Marker::MODIFY;
    geometry_msgs::Point p;
    direction_marker.points.push_back(p);
    p.x = std::cos(yaw);
    p.y = -std::sin(yaw);
    direction_marker.points.push_back(p);
    direction_marker.color.a = 1;
    direction_marker.color.r = 0;
    direction_marker.color.g = 1;
    direction_marker.color.b = 1;
    direction_marker.scale.x = 0.05;
    direction_marker.scale.y = 0.08;
    if (debug_settings_.debug_direction) {
        debug_direction_marker_pub_.publish(direction_marker);
    }

    // Cluster lines by orientation
    std::vector<Eigen::Vector3d> para_line_normals;
    std::vector<Eigen::Vector3d> perp_line_normals;
    splitLinesByOrientation(best_theta,
                            pl_normals,
                            para_line_normals,
                            perp_line_normals);

    // Convert lines to distances from origin
    std::vector<double> para_signed_dists;
    std::vector<double> perp_signed_dists;
    getUnAltifiedDistancesFromLines(best_theta,
                                    para_line_normals,
                                    perp_line_normals,
                                    para_signed_dists,
                                    perp_signed_dists);

    // Extract altitude
    // TODO

    // Extract horizontal translation
    const Eigen::Vector2d last_position_2d {
        last_filtered_position_(0),
        last_filtered_position_(1)
    };

    Eigen::Vector2d position_2d;
    Eigen::Matrix2d position_cov_2d;
    get2dPosition(para_signed_dists,
                  perp_signed_dists,
                  best_theta,
                  height,
                  last_position_2d,
                  position_2d,
                  position_cov_2d);

    // Publish updated position
    geometry_msgs::TransformStamped camera_to_lq_transform;
    if (!transform_wrapper_.getTransformAtTime(camera_to_lq_transform,
                                                     "level_quad",
                                                     "bottom_camera_optical",
                                                     time,
                                                     ros::Duration(1.0))) {
        throw ros::Exception(
            "Failed to get transform from level_quad to bottom_camera_optical");
    }
    geometry_msgs::PointStamped camera_position;
    camera_position.header.frame_id = "bottom_camera_optical";
    camera_position.point.x = 0;
    camera_position.point.y = 0;
    camera_position.point.z = 0;
    tf2::doTransform(camera_position, camera_position, camera_to_lq_transform);

    ROS_DEBUG("px %f py %f pz %f", position_2d(0), position_2d(1), position_2d(2));
    ROS_DEBUG("height %f", height);

    Eigen::Vector3d position_3d {
        position_2d(0) - camera_position.point.x,
        position_2d(1) - camera_position.point.y,
        height         - camera_position.point.z
    };

    Eigen::Matrix3d position_cov_3d = Eigen::Matrix3d::Zero();
    position_cov_3d.block<2, 2>(0, 0) = position_cov_2d;

    publishPositionEstimate(position_3d, position_cov_3d, time);
}

void GridLineEstimator::publishPositionEstimate(
        const Eigen::Vector3d& position,
        const Eigen::Matrix3d& covariance,
        const ros::Time& time) const
{
    geometry_msgs::PoseWithCovarianceStamped pose_msg;

    pose_msg.header.stamp = time;
    pose_msg.header.frame_id = "map";

    pose_msg.pose.pose.position.x = position(0);
    pose_msg.pose.pose.position.y = position(1);
    pose_msg.pose.pose.position.z = position(2);

    if (!std::isfinite(position(0))
     || !std::isfinite(position(1))
     || !std::isfinite(position(2))) {
        throw ros::Exception("Position was not finite");
    }

    pose_msg.pose.pose.orientation.w = 1;
    pose_msg.pose.pose.orientation.x = 0;
    pose_msg.pose.pose.orientation.y = 0;
    pose_msg.pose.pose.orientation.z = 0;

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            pose_msg.pose.covariance[6*i + j] = covariance(i, j);
        }
    }

    pose_pub_.publish(pose_msg);
}

void GridLineEstimator::publishYaw(double yaw, const ros::Time& time) const
{
    iarc7_msgs::Float64Stamped msg;
    msg.header.stamp = time;
    msg.data = yaw;
    yaw_pub_.publish(msg);
}

void GridLineEstimator::splitLinesByOrientation(
        double theta,
        const std::vector<Eigen::Vector3d>& pl_normals,
        std::vector<Eigen::Vector3d>& para_line_normals,
        std::vector<Eigen::Vector3d>& perp_line_normals) const
{
    para_line_normals.clear();
    perp_line_normals.clear();

    std::ostringstream thetas_stream;

    const double angle_thresh
        = grid_estimator_settings_.line_rejection_angle_threshold;

    for (const Eigen::Vector3d& pl_normal : pl_normals) {
        thetas_stream << std::atan(-pl_normal(0)/pl_normal(1)) << " ";
        double dist_to_line = std::abs(theta
                                     - std::atan(-pl_normal(0)/pl_normal(1)));
        if (dist_to_line < angle_thresh || dist_to_line > M_PI - angle_thresh) {
            para_line_normals.push_back(pl_normal);
        } else if (std::abs(dist_to_line - M_PI/2) < angle_thresh) {
            perp_line_normals.push_back(pl_normal);
        } else {
            ROS_WARN("Throwing out line with angle %f when we're at angle %f",
                     std::atan(-pl_normal(0)/pl_normal(1)),
                     theta);
        }
    }

    ROS_DEBUG_STREAM("GridLineEstimator thetas: (" << thetas_stream.str()
                  << ") Extracted theta: (" << theta << ")");
}

void GridLineEstimator::updateFilteredPosition(const ros::Time& time)
{
    geometry_msgs::TransformStamped filtered_position_transform_stamped;
    if (!transform_wrapper_.getTransformAtTime(
            filtered_position_transform_stamped,
            "map",
            "bottom_camera_optical",
            time,
            ros::Duration(1.0))) {
        ROS_ERROR("Failed to fetch transform to bottom_camera_optical");
    } else {
        geometry_msgs::PointStamped camera_position;
        tf2::doTransform(camera_position,
                         camera_position,
                         filtered_position_transform_stamped);

        last_filtered_position_(0) = camera_position.point.x;
        last_filtered_position_(1) = camera_position.point.y;
        last_filtered_position_(2) = std::isfinite(debug_settings_.debug_height)
                                   ? debug_settings_.debug_height
                                   : camera_position.point.z;
        last_filtered_position_stamp_ = time;
    }
}

} // namespace iarc7_vision
