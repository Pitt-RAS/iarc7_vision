#include "iarc7_vision/RoombaGHT.hpp"

#include <chrono>

#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// BAD HEADERS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
// END BAD HEADERS

namespace iarc7_vision
{

RoombaGHT::RoombaGHT(const RoombaEstimatorSettings& settings,
                     ros::NodeHandle& private_nh)
    : private_nh_(private_nh),
      ght_(cv::cuda::createGeneralizedHoughBallard()),
      settings_(settings),
      debug_edges_pub_(private_nh_.advertise<sensor_msgs::Image>("edges", 10))
{
    double min_dist = settings_.template_pixels_per_meter
                    * settings_.roomba_plate_width;
    ght_->setMinDist(min_dist);

    //ght_->setMinScale(0.8);
    //ght_->setMaxScale(1.2);

    ght_->setMaxBufferSize(1000); // maximal size of inner buffers
    //ght_->setMinAngle(0);
    //ght_->setMaxAngle(360);

    onSettingsChanged();
}

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

bool getThetaForThetas(
        const cv::Mat& dx,
        const cv::Mat& dy,
        const cv::Mat& points,
        double& theta)
{
    double theta_step = 0.1;
    // extract angles from lines (angles in the list are in [0, pi/2))
    std::vector<double> thetas;

    // need to optimize on gpu?  at least optimize for cache?
    for (int i = 0; i < points.size().width; i++) {
        for (int j = 0; j < points.size().height; j++) {
            if (points.at<uchar>(j, i) == 255) {
                double theta = std::atan2(dy.at<int32_t>(j, i), dx.at<int32_t>(j, i));
                theta = std::fmod(theta + M_PI, M_PI/2);
                thetas.push_back(theta);
            }
        }
    }

    if (thetas.size() == 0) {
        ROS_ERROR("No thetas");
        return false;
    }

    // do a coarse sweep over angles from 0 to pi/2
    double best_coarse_theta = 0;
    double best_coarse_theta_score = theta_loss(thetas, 0);
    for (double theta = theta_step;
         theta < M_PI / 2;
         theta += theta_step) {
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

    theta = best_theta;
    return true;
}

bool RoombaGHT::detect(const cv::cuda::GpuMat& image,
                       const cv::Rect& bounding_rect,
                       cv::Point2f& pos,
                       double& angle)
{
    // First grab the important area of the image
    cv::cuda::GpuMat image_crop(image, bounding_rect);

    // TRY HSV
    cv::cuda::GpuMat hsv;
    cv::cuda::cvtColor(image_crop, hsv, CV_BGR2HSV);
    cv::cuda::GpuMat hsv_channels[3];
    cv::cuda::split(hsv, hsv_channels);

    // Run the GHT, there should only be one found

    auto canny = cv::cuda::createCannyEdgeDetector(
            settings_.ght_canny_low_thresh,
            settings_.ght_canny_high_thresh);
    auto filterdx = cv::cuda::createSobelFilter(CV_8UC1, CV_32S, 1, 0);
    auto filterdy = cv::cuda::createSobelFilter(CV_8UC1, CV_32S, 0, 1);
    cv::cuda::GpuMat dx, dy, edges;
    filterdx->apply(hsv_channels[1], dx);
    filterdy->apply(hsv_channels[1], dy);
    canny->detect(dx, dy, edges);

    cv::Mat dx_c, dy_c, edges_c;
    dx.download(dx_c);
    dy.download(dy_c);
    edges.download(edges_c);

    double theta;
    double success = getThetaForThetas(dx_c, dy_c, edges_c, theta);
    if (!success) return false;

    // ROTATE
    int diag = std::hypot(dx.size().width, dx.size().height);

    cv::Vec4f best_pos;
    int best_i;
    int max_votes = 0;
    for (int i = 0; i < 4; i++) {
        cv::Mat rotated_image_1 = cv::Mat::zeros(diag, diag, hsv_channels[1].type());

        cv::Rect roi ((diag - dx.size().width) / 2,
                      (diag - dy.size().height) / 2,
                      dx.size().width,
                      dx.size().height);
        cv::Mat saturation;
        hsv_channels[1].download(saturation);
        saturation.copyTo(rotated_image_1(roi));

        rotated_image_1.convertTo(rotated_image_1, CV_32FC1);

        ROS_ERROR("theta: %f", theta);
        const auto rotation = cv::getRotationMatrix2D(
                cv::Point2f(diag / 2, diag / 2),
                theta * 180 / M_PI + 90*i,
                1);

        cv::Mat rotated_image = cv::Mat::zeros(diag, diag, rotated_image_1.type());
        cv::warpAffine(rotated_image_1, rotated_image, rotation, rotated_image.size());

        rotated_image.convertTo(rotated_image, CV_8UC1);

        cv::Mat rotated_dx, rotated_dy, rotated_edges;
        cv::cuda::GpuMat rotated_image_d(rotated_image);
        cv::cuda::GpuMat rotated_dx_d, rotated_dy_d, rotated_edges_d;
        filterdx->apply(rotated_image_d, rotated_dx_d);
        filterdy->apply(rotated_image_d, rotated_dy_d);
        canny->detect(rotated_dx_d, rotated_dy_d, rotated_edges_d);
        rotated_dx_d.download(rotated_dx);
        rotated_dy_d.download(rotated_dy);
        rotated_edges_d.download(rotated_edges);

        auto canny_ght = cv::cuda::createCannyEdgeDetector(
                settings_.ght_canny_low_thresh,
                settings_.ght_canny_high_thresh);
        auto filterdx_ght = cv::cuda::createSobelFilter(CV_8UC1, CV_32F, 1, 0);
        auto filterdy_ght = cv::cuda::createSobelFilter(CV_8UC1, CV_32F, 0, 1);
        cv::cuda::GpuMat ght_dx, ght_dy, ght_edges;
        filterdx_ght->apply(rotated_image_d, ght_dx);
        filterdy_ght->apply(rotated_image_d, ght_dy);
        canny_ght->detect(rotated_image_d, ght_edges);

        //cv::Mat pub_edges;
        //ght_edges.download(pub_edges);
        //const cv_bridge::CvImage cv_image {
        //    std_msgs::Header(),
        //    sensor_msgs::image_encodings::MONO8,
        //    pub_edges
        //};

        //debug_edges_pub_.publish(cv_image.toImageMsg());

        cv::cuda::GpuMat gpu_position, gpu_votes;
        const auto start = std::chrono::high_resolution_clock::now();
        ght_->detect(ght_edges, ght_dx, ght_dy, gpu_position, gpu_votes);
        const auto end = std::chrono::high_resolution_clock::now();
        ROS_ERROR_STREAM("Time: "
                      << std::chrono::duration_cast<std::chrono::microseconds>(
                            end - start).count());

        if (gpu_position.empty()) continue;

        std::vector<cv::Vec4f> position;
        std::vector<cv::Vec3i> votes;
        gpu_position.download(position);
        gpu_votes.download(votes);

        if (votes[0][0] > max_votes) {
            ROS_ERROR("Better pos: %d %d %d", votes[0][0], max_votes, i);
            max_votes = votes[0][0];
            best_i = i;
            best_pos = position[0];
        }
    }

    if (max_votes == 0) {
        ROS_ERROR("NO GHT FOUND");
        return false;
    }

    // ROTATE AND PUBLISH
    cv::Mat rotated_image_1 = cv::Mat::zeros(diag, diag, hsv_channels[1].type());

    cv::Rect roi ((diag - dx.size().width) / 2,
                  (diag - dy.size().height) / 2,
                  dx.size().width,
                  dx.size().height);
    cv::Mat saturation;
    hsv_channels[1].download(saturation);
    saturation.copyTo(rotated_image_1(roi));

    rotated_image_1.convertTo(rotated_image_1, CV_32FC1);

    const auto rotation = cv::getRotationMatrix2D(
            cv::Point2f(diag / 2, diag / 2),
            theta * 180 / M_PI + 90*best_i,
            1);

    cv::Mat rotated_image = cv::Mat::zeros(diag, diag, rotated_image_1.type());
    cv::warpAffine(rotated_image_1, rotated_image, rotation, rotated_image.size());

    rotated_image.convertTo(rotated_image, CV_8UC1);

    const cv_bridge::CvImage cv_image {
        std_msgs::Header(),
        sensor_msgs::image_encodings::MONO8,
        rotated_image
    };

    debug_edges_pub_.publish(cv_image.toImageMsg());
    //////////////////////////////

    pos.x = bounding_rect.x + best_pos[0];
    pos.y = bounding_rect.y + best_pos[1];
    angle = theta + M_PI/2*best_i;
    return true;
}

void RoombaGHT::onSettingsChanged()
{
    ght_ = cv::cuda::createGeneralizedHoughBallard();

    double min_dist = settings_.template_pixels_per_meter
                    * settings_.roomba_plate_width;
    ght_->setMinDist(min_dist);

    //ght_->setMinScale(0.8);
    //ght_->setMaxScale(1.2);

    ght_->setMaxBufferSize(1000); // maximal size of inner buffers
    //ght_->setMinAngle(0);
    //ght_->setMaxAngle(360);

    ght_->setVotesThreshold(settings_.ght_pos_thresh);
    ght_->setCannyLowThresh(settings_.ght_canny_low_thresh);
    ght_->setCannyHighThresh(settings_.ght_canny_high_thresh);
    ght_->setDp(settings_.ght_dp);
    ght_->setLevels(settings_.ght_levels);

    cv::Mat templ = cv::imread("roomba_template.png", cv::IMREAD_GRAYSCALE);
    cv::cuda::GpuMat gpu_templ(templ);
    ght_->setTemplate(gpu_templ);
}

} // namespace iarc7_vision
