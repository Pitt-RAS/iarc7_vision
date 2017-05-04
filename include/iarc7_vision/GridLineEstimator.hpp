#ifndef IARC7_VISION_GRIDLINE_ESTIMATOR_HPP_
#define IARC7_VISION_GRIDLINE_ESTIMATOR_HPP_

// BAD HEADERS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Geometry>
#pragma GCC diagnostic pop
// END BAD HEADERS

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros_utils/SafeTransformWrapper.hpp>
#include <tf2_ros/transform_listener.h>

namespace iarc7_vision {

struct LineExtractorSettings {
    double canny_low_threshold;
    double canny_high_threshold;
    int canny_sobel_size;
    double hough_rho_resolution;
    double hough_theta_resolution;
    double hough_thresh_fraction;
};

struct GridEstimatorSettings {
    double theta_step;
    double grid_step;
    double grid_spacing;
    double grid_line_thickness;
    Eigen::Vector2d grid_zero_offset;
    int grid_translation_mean_iterations;
};

struct GridLineDebugSettings {
    bool debug_edges;
    bool debug_lines;
};

class GridLineEstimator {
  public:
    GridLineEstimator(double fov,
                      const LineExtractorSettings& line_estimator_settings,
                      const GridEstimatorSettings& grid_estimator_settings,
                      const GridLineDebugSettings& debug_settings);
    void update(const cv::Mat& image, const ros::Time& time);

  private:

    double getCurrentTheta(const ros::Time& time) const;

    /// Compute the focal length (in px) from image size and dfov
    ///
    /// @param[in] fov Field of view in radians
    static double getFocalLength(const cv::Size& img_size, double fov);

    /// Return all the lines in the image
    ///
    /// @param[out] lines  {Lines in the form (rho, theta), where theta is the
    ///                     angle (from right, positive is down) and rho is the
    ///                     distance (in pixels) from the center of the image
    ///                     (NOTE: rho is not the same as in OpenCV HoughLines)}
    /// @param[in]  image  Input image
    /// @param[in]  height Approximate altitude of the camera, in meters
    ///
    /// TODO: change units of height parameter to pixels
    void getLines(std::vector<cv::Vec2f>& lines,
                  const cv::Mat& image,
                  double height) const;

    /// each line that the camera sees defines a plane (call it P_l) in which
    /// the actual line resides. We can find the line in the floor plane
    /// (call the line l_f and the plane P_f) by intersecting P_f with P_l.
    ///
    ///
    /// the equation for P_l is then pl_normal * v = pl_normal.z * z_cam,
    /// where v is a vector in P_l and z_cam is the distance from the ground
    /// to the camera.
    ///
    /// this means the equation for l_f (in the map frame) is
    /// pl_normal.x * x + pl_normal.y * y = pl_normal.z * z_cam
    ///
    /// because z_cam is the same for all lines, each line is specified by
    /// pl_normal alone
    void getPlanesForImageLines(const std::vector<cv::Vec2f>& image_lines,
                                const ros::Time& time,
                                double focal_length,
                                std::vector<Eigen::Vector3d>& pl_normals) const;

    double getThetaForPlanes(const std::vector<Eigen::Vector3d>& pl_normals) const;

    /// Returns signed distance of each line from the origin without accounting
    /// for altitude.  To the actual signed distance, multiply by the distance
    /// from the camera frame origin to the ground.
    ///
    /// This operates on the assumption that we can use the distances between
    /// these lines and the origin (based on a projection onto the theta vector)
    /// in the map frame (the frame the lines are in) as a good estimate of the
    /// translation of the grid.  This is true if the camera is pointed
    /// straight downward, but if the camera is close to horizontal, small
    /// errors in line angle create large errors in distance between the line
    /// and the origin.  One possible solution would be to project the camera
    /// forward vector into the ground plane and use distances from that point
    /// instead of from the origin.
    static void getUnAltifiedDistancesFromLines(
        double theta,
        const std::vector<Eigen::Vector3d>& para_line_normals,
        const std::vector<Eigen::Vector3d>& perp_line_normals,
        std::vector<double>& para_signed_dists,
        std::vector<double>& perp_signed_dists);

    void get1dGridShift(const std::vector<double>& wrapped_dists,
                        double& value,
                        double& variance) const;

    /// @param[in] position_estimate Vector from origin of map to origin of pl_normal frame ("level_quad" on the ground)
    void get2dPosition(const std::vector<double>& para_signed_dists,
                       const std::vector<double>& perp_signed_dists,
                       double theta,
                       double height_estimate,
                       const Eigen::Vector2d& position_estimate,
                       Eigen::Vector2d& position,
                       Eigen::Matrix2d& covariance) const;

    double gridLoss(const std::vector<double>& wrapped_dists,
                    double dist) const;

    void processImage(const cv::Mat& image, const ros::Time& time) const;

    /// Publish a 3d position estimate with the specified timestamp in the
    /// "map" frame
    void publishPositionEstimate(
        const Eigen::Vector3d& position,
        const Eigen::Matrix3d& covariance,
        const ros::Time& time) const;

    /// Takes a list of plane normals and splits them into two clusters based
    /// on which are parallel or perpendicular to the theta vector.  The theta
    /// vector is defined (in the frame of pl_normals) as
    /// [cos(theta), sin(theta), 0]
    ///
    /// @param[in]  theta             {Orientation of the grid in the same
    ///                                frame as pl_normals, in [0, pi/2)}
    /// @param[in]  pl_normals        {Plane normals (see implementation of
    ///                                update)}
    /// @param[out] para_line_normals {Plane normals of lines closer to
    ///                                parallel to the theta vector}
    /// @param[out] perp_line_normals {Plane normals of lines closer to
    ///                                perpendicular to the theta vector}
    static void splitLinesByOrientation(
        double theta,
        const std::vector<Eigen::Vector3d>& pl_normals,
        std::vector<Eigen::Vector3d>& para_line_normals,
        std::vector<Eigen::Vector3d>& perp_line_normals);

    ros::Publisher pose_pub_;

    mutable cv::Mat image_sized_;
    mutable cv::Mat image_hsv_;
    mutable cv::Mat image_edges_;
    mutable cv::Mat image_hsv_channels_[3];

    mutable cv::gpu::HoughLinesBuf gpu_hough_buf_;
    mutable cv::gpu::GpuMat gpu_image_;
    mutable cv::gpu::GpuMat gpu_image_sized_;
    mutable cv::gpu::GpuMat gpu_image_hsv_;
    mutable cv::gpu::GpuMat gpu_image_edges_;
    mutable cv::gpu::GpuMat gpu_lines_;
    mutable cv::gpu::GpuMat gpu_image_hsv_channels_[3];

    const double fov_; // diagonal field of view of our bottom camera

    const LineExtractorSettings& line_extractor_settings_;
    const GridEstimatorSettings& grid_estimator_settings_;

    const GridLineDebugSettings& debug_settings_;
    ros::Publisher debug_edges_pub_;
    ros::Publisher debug_lines_pub_;

    Eigen::Vector3d last_filtered_position_;

    ros_utils::SafeTransformWrapper transform_wrapper_;
};

} // namespace iarc7_vision

#endif // include guard
