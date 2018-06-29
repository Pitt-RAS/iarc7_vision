#include "iarc7_vision/UndistortionModel.hpp"

#include "ros_utils/ParamUtils.hpp"

namespace iarc7_vision {

UndistortionModel::UndistortionModel(const ros::NodeHandle& nh,
                                     const cv::Size& image_size)
    : image_size_(image_size)
{
    cv::Mat camera_matrix(3, 3, CV_32FC1);
    camera_matrix.at<float>(0, 0) =
        ros_utils::ParamUtils::getParam<double>(nh, "f_x");
    camera_matrix.at<float>(0, 1) = 0.;
    camera_matrix.at<float>(0, 2) =
        ros_utils::ParamUtils::getParam<double>(nh, "c_x");
    camera_matrix.at<float>(1, 0) = 0.;
    camera_matrix.at<float>(1, 1) =
        ros_utils::ParamUtils::getParam<double>(nh, "f_y");
    camera_matrix.at<float>(1, 2) =
        ros_utils::ParamUtils::getParam<double>(nh, "c_y");
    camera_matrix.at<float>(2, 0) = 0.;
    camera_matrix.at<float>(2, 1) = 0.;
    camera_matrix.at<float>(2, 2) = 1.;

    cv::Mat dist(5, 1, CV_32FC1);
    dist.at<float>(0, 0) = ros_utils::ParamUtils::getParam<double>(nh, "k1");
    dist.at<float>(1, 0) = ros_utils::ParamUtils::getParam<double>(nh, "k2");
    dist.at<float>(2, 0) = ros_utils::ParamUtils::getParam<double>(nh, "p1");
    dist.at<float>(3, 0) = ros_utils::ParamUtils::getParam<double>(nh, "p2");
    dist.at<float>(4, 0) = ros_utils::ParamUtils::getParam<double>(nh, "k3");

    cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(
            camera_matrix, dist, image_size_, 0);

    cv::Mat map1_cpu;
    cv::Mat map2_cpu;
    cv::initUndistortRectifyMap(camera_matrix,
                                dist,
                                cv::Mat(),
                                new_camera_matrix,
                                image_size_,
                                CV_32FC1,
                                map1_cpu,
                                map2_cpu);

    map1_ = cv::cuda::GpuMat(map1_cpu);
    map2_ = cv::cuda::GpuMat(map2_cpu);
}

void UndistortionModel::undistort(const cv::cuda::GpuMat& in,
                                  cv::cuda::GpuMat& out) const
{
    if (in.size() != image_size_) {
        throw std::runtime_error("Image size does not match");
    }

    cv::cuda::remap(in, out, map1_, map2_, cv::INTER_LINEAR);
}

} // namespace iarc7_vision
