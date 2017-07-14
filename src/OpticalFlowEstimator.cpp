#include "iarc7_vision/OpticalFlowEstimator.hpp"

// BAD HEADER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cv_bridge/cv_bridge.h>
#pragma GCC diagnostic pop
// END BAD HEADER

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <ros_utils/SafeTransformWrapper.hpp>

#include <geometry_msgs/Vector3Stamped.h>
#include <iarc7_msgs/Float64Stamped.h>
#include <visualization_msgs/Marker.h>

namespace iarc7_vision {

static void download(const cv::gpu::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const cv::gpu::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

static void drawArrows(cv::Mat& frame, const std::vector<cv::Point2f>& prevPts, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status, cv::Scalar line_color = cv::Scalar(0, 0, 255))
{
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            int line_thickness = 1;

            cv::Point p = prevPts[i];
            cv::Point q = nextPts[i];

            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);

            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );

            if (hypotenuse < 1.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

OpticalFlowEstimator::OpticalFlowEstimator(
        const OpticalFlowEstimatorSettings& flow_estimator_settings,
        const OpticalFlowDebugSettings& debug_settings)
    : flow_estimator_settings_(flow_estimator_settings),
      debug_settings_(debug_settings),
      last_filtered_position_(),
      transform_wrapper_(),
      last_message_(),
      last_scaled_image_(),
      last_scaled_grayscale_image_(),
      debug_velocity_vector_image_pub_()
{
    ros::NodeHandle local_nh ("optical_flow_estimator");

    if (debug_settings_.debug_vectors_image) {
        debug_velocity_vector_image_pub_
            = local_nh.advertise<sensor_msgs::Image>("vector_image", 1);
    }

    /*if (debug_settings_.debug_edges) {
        debug_edges_pub_ = local_nh.advertise<sensor_msgs::Image>("edges", 10);
    }

    if (debug_settings_.debug_lines) {
        debug_lines_pub_ = local_nh.advertise<sensor_msgs::Image>("lines", 10);
    }

    if (debug_settings_.debug_line_markers) {
        debug_line_markers_pub_ = local_nh.advertise<visualization_msgs::Marker>(
                "line_markers", 1);
    }*/

    twist_pub_
        = local_nh.advertise<geometry_msgs::TwistWithCovarianceStamped>("twist",
                                                                       10);

}

void OpticalFlowEstimator::update(const sensor_msgs::Image::ConstPtr& message)
{
    int64 start = cv::getTickCount();
    updateFilteredPosition(message->header.stamp);
    double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "updateFilteredPosition : " << timeSec << " sec" << std::endl;

    if (last_message_!=nullptr) {
        if (last_filtered_position_.point.z
                >= flow_estimator_settings_.min_estimation_altitude) {
            try {

                cv::Mat last_image = cv_bridge::toCvShare(last_message_)->image;
                cv::Mat curr_image = cv_bridge::toCvShare(message)->image;

                geometry_msgs::TwistWithCovarianceStamped velocity;
                timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
                std::cout << "pre estimateVelocity : " << timeSec << " sec" << std::endl;
                estimateVelocity(velocity, last_image, curr_image, last_filtered_position_.point.z);
                timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
                std::cout << "post estimateVelocity : " << timeSec << " sec" << std::endl;
            } catch (const std::exception& ex) {
                ROS_ERROR_STREAM("Caught exception processing image flow: "
                              << ex.what());
            }
        } else {
            ROS_WARN("Height (%f) is below min processing height (%f)",
                     last_filtered_position_.point.z,
                     flow_estimator_settings_.min_estimation_altitude);
        }
        last_message_ = message;
    }
    else {

        last_message_ = message;

        cv::Mat image = cv_bridge::toCvShare(message)->image;

        cv::Size image_size = image.size();
        image_size.width = image_size.width * flow_estimator_settings_.scale_factor;
        image_size.height = image_size.height * flow_estimator_settings_.scale_factor;

        cv::gpu::GpuMat d_frame1_big(image);
        cv::gpu::GpuMat d_frame1;
        cv::gpu::GpuMat d_frame1Gray;

        cv::gpu::resize(d_frame1_big,
                        d_frame1,
                        image_size);

        cv::gpu::cvtColor(d_frame1,
                          d_frame1Gray,
                          CV_RGBA2GRAY);

        //d_frame1.copyTo(last_scaled_image_);
        //d_frame1Gray.copyTo(last_scaled_grayscale_image_);
        last_scaled_image_ = d_frame1;
        last_scaled_grayscale_image_ = d_frame1Gray;
    }
    timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "update return : " << timeSec << " sec" << std::endl;
}

double OpticalFlowEstimator::getFocalLength(const cv::Size& img_size, double fov)
{
    return std::hypot(img_size.width/2.0, img_size.height/2.0)
         / std::tan(fov / 2.0);
}

void OpticalFlowEstimator::estimateVelocity(geometry_msgs::TwistWithCovarianceStamped&,
                                            const cv::Mat&,
                                            const cv::Mat& image,
                                            double)
{
    // m/px = camera_height / focal_length;
    //double current_meters_per_px = height
    //                     / getFocalLength(image.size(),
    //                                      flow_estimator_settings_.fov);

    // desired m_px used to keep kernel sizes relative to our features
    //double desired_meters_per_px = 1.0
    //                             / flow_estimator_settings_.pixels_per_meter;

    //double scale_factor = current_meters_per_px / desired_meters_per_px;

    //cv::Mat image_edges;

    cv::Size image_size = image.size();
    image_size.width = image_size.width * flow_estimator_settings_.scale_factor;
    image_size.height = image_size.height * flow_estimator_settings_.scale_factor;

    if (cv::gpu::getCudaEnabledDeviceCount() == 0) {
        ROS_ERROR_ONCE("Optical Flow Estimator does not have a CPU version");

    } else {

        // goodFeaturesToTrack

        int64 start = cv::getTickCount();

        cv::gpu::GoodFeaturesToTrackDetector_GPU detector(
                        flow_estimator_settings_.points,
                        flow_estimator_settings_.quality_level,
                        flow_estimator_settings_.min_dist);

        double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "create good features to track : " << timeSec << " sec" << std::endl;

        cv::gpu::GpuMat d_prevPts;

        cv::gpu::GpuMat d_frame1_big(image);
        cv::gpu::GpuMat d_frame1;
        cv::gpu::GpuMat d_frame1Gray;

        cv::gpu::resize(d_frame1_big,
                        d_frame1,
                        image_size);

        timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "post resize : " << timeSec << " sec" << std::endl;

        cv::gpu::cvtColor(d_frame1,
                          d_frame1Gray,
                          CV_RGBA2GRAY);

        timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "post cvtColor : " << timeSec << " sec" << std::endl;

        detector(last_scaled_grayscale_image_, d_prevPts);

        timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "post detector : " << timeSec << " sec" << std::endl;

        // Sparse

        cv::gpu::PyrLKOpticalFlow d_pyrLK;

        d_pyrLK.winSize.width = flow_estimator_settings_.win_size;
        d_pyrLK.winSize.height = flow_estimator_settings_.win_size;
        d_pyrLK.maxLevel = flow_estimator_settings_.max_level;
        d_pyrLK.iters = flow_estimator_settings_.iters;

        cv::gpu::GpuMat d_nextPts;
        cv::gpu::GpuMat d_status;

        d_pyrLK.sparse(last_scaled_image_, d_frame1, d_prevPts, d_nextPts, d_status);
        timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "PYRLK sparse : " << timeSec << " sec" << std::endl;

        last_scaled_image_ = d_frame1;
        last_scaled_grayscale_image_ = d_frame1Gray;

        //last_scaled_image_ = d_frame1.clone();
        //last_scaled_grayscale_image_ = d_frame1Gray.clone();

        //d_frame1.assignTo(last_scaled_image_);
        //last_scaled_grayscale_image_(d_frame1Gray);

        //d_frame1.copyTo(last_scaled_image_);
        //d_frame1Gray.copyTo(last_scaled_grayscale_image_);

        if (debug_settings_.debug_vectors_image) {

            // Draw arrows

            std::vector<cv::Point2f> prevPts(d_prevPts.cols);
            download(d_prevPts, prevPts);

            std::vector<cv::Point2f> nextPts(d_nextPts.cols);
            download(d_nextPts, nextPts);

            std::vector<uchar> status(d_status.cols);
            download(d_status, status);

            cv::Mat temp;
            d_frame1.download(temp);

            drawArrows(temp, prevPts, nextPts, status, cv::Scalar(255, 0, 0));

            cv_bridge::CvImage cv_image {
                std_msgs::Header(),
                sensor_msgs::image_encodings::RGBA8,
                temp
            };

            debug_velocity_vector_image_pub_.publish(cv_image.toImageMsg());
        }

        timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "post debug vectors image : " << timeSec << " sec" << std::endl;

        //cv::imshow("PyrLK [Sparse]", temp);
        //cv::waitKey(10);
    }
}

void OpticalFlowEstimator::updateFilteredPosition(const ros::Time& time)
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

        last_filtered_position_ = camera_position;
    }
}

} // namespace iarc7_vision
