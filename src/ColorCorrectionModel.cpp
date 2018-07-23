#include "iarc7_vision/ColorCorrectionModel.hpp"

#include <chrono>

#include "ros_utils/ParamUtils.hpp"

namespace iarc7_vision {

ColorCorrectionModel::ColorCorrectionModel(const ros::NodeHandle& nh)
    : a00_(ros_utils::ParamUtils::getParam<double>(nh, "a00")),
      a01_(ros_utils::ParamUtils::getParam<double>(nh, "a01")),
      a02_(ros_utils::ParamUtils::getParam<double>(nh, "a02")),
      a10_(ros_utils::ParamUtils::getParam<double>(nh, "a10")),
      a11_(ros_utils::ParamUtils::getParam<double>(nh, "a11")),
      a12_(ros_utils::ParamUtils::getParam<double>(nh, "a12")),
      a20_(ros_utils::ParamUtils::getParam<double>(nh, "a20")),
      a21_(ros_utils::ParamUtils::getParam<double>(nh, "a21")),
      a22_(ros_utils::ParamUtils::getParam<double>(nh, "a22")),
      offset0_(ros_utils::ParamUtils::getParam<double>(nh, "offset0")),
      offset1_(ros_utils::ParamUtils::getParam<double>(nh, "offset1")),
      offset2_(ros_utils::ParamUtils::getParam<double>(nh, "offset2")),
      gamma_(ros_utils::ParamUtils::getParam<double>(nh, "gamma")),
      final_gamma_(ros_utils::ParamUtils::getParam<double>(nh, "final_gamma"))
{
    {
        cv::Mat lut(cv::Size(256, 1), CV_8UC3);
        for (int i = 0; i < 256; i++) {
            const double result = 255. * std::pow(static_cast<double>(i) / 255.,
                                           1/gamma_);
            const uchar result_b = std::max(0., std::min(255., result));
            lut.at<cv::Vec3b>(0, i) = cv::Vec3b(result_b, result_b, result_b);
        }
        gamma_lut_ = cv::cuda::createLookUpTable(lut);
    }

    {
        cv::Mat lut(cv::Size(256, 1), CV_8UC3);
        for (int i = 0; i < 256; i++) {
            const double result = 255. * std::pow(static_cast<double>(i) / 255.,
                                           1/final_gamma_);
            const uchar result_b = std::max(0., std::min(255., result));
            lut.at<cv::Vec3b>(0, i) = cv::Vec3b(result_b, result_b, result_b);
        }
        final_lut_ = cv::cuda::createLookUpTable(lut);
    }
}

void ColorCorrectionModel::correct(const cv::cuda::GpuMat& in,
                                   cv::cuda::GpuMat& out,
                                   cv::cuda::Stream& stream) const
{
    const auto start = std::chrono::high_resolution_clock::now();

    gamma_lut_->transform(in, in_post_gamma_, stream);

    const auto after_gamma = std::chrono::high_resolution_clock::now();

    cv::cuda::split(in_post_gamma_, in_channels_.data(), stream);

    for (int i = 0; i < 3; i++) {
        cv::cuda::compare(in_channels_[i], 255, saturated_masks_[i], cv::CMP_GE, stream);
    }

    std::array<cv::cuda::GpuMat, 3> gbr_channels = {{in_channels_[1],
                                                     in_channels_[2],
                                                     in_channels_[0]}};
    cv::cuda::merge(gbr_channels.data(), 3, gbr_order_, stream);

    std::array<cv::cuda::GpuMat, 3> brg_channels = {{in_channels_[2],
                                                     in_channels_[0],
                                                     in_channels_[1]}};
    cv::cuda::merge(brg_channels.data(), 3, brg_order_, stream);

    const auto after_splits = std::chrono::high_resolution_clock::now();

    cv::cuda::multiply(in_post_gamma_,
                       cv::Scalar(a00_, a11_, a22_),
                       out1_,
                       1.,
                       CV_32FC3,
                       stream);
    cv::cuda::multiply(gbr_order_,
                       cv::Scalar(a01_, a12_, a20_),
                       out2_,
                       1.,
                       CV_32FC3,
                       stream);
    cv::cuda::multiply(brg_order_,
                       cv::Scalar(a02_, a10_, a21_),
                       out3_,
                       1.,
                       CV_32FC3,
                       stream);

    cv::cuda::add(out1_, out2_, out_float1_, cv::noArray(), CV_32FC3, stream);
    cv::cuda::add(out3_,
                  cv::Scalar(offset0_, offset1_, offset2_),
                  out_float2_,
                  cv::noArray(),
                  CV_32FC3,
                  stream);

    cv::cuda::add(out_float1_, out_float2_, out_float_, cv::noArray(), CV_32FC3, stream);

    const auto after_arithm = std::chrono::high_resolution_clock::now();

    out_float_.convertTo(out_before_gamma_, CV_8UC3);
    final_lut_->transform(out_before_gamma_, out_after_gamma_, stream);

    const auto after_gamma2 = std::chrono::high_resolution_clock::now();

    cv::cuda::split(out_after_gamma_, out_channels_.data(), stream);
    for (int i = 0; i < 3; i++) {
        out_channels_[i].setTo(255, saturated_masks_[i], stream);
    }
    cv::cuda::merge(out_channels_.data(), 3, out, stream);

    const auto after_out = std::chrono::high_resolution_clock::now();

    const auto counts = [](const auto& a, const auto& b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                b - a).count();
    };
    ROS_DEBUG_STREAM(
            "Gamma: " << counts(start, after_gamma) << std::endl
         << "Splits: " << counts(after_gamma, after_splits) << std::endl
         << "Arithm: " << counts(after_splits, after_arithm) << std::endl
         << "Gamma2: " << counts(after_arithm, after_gamma2) << std::endl
         << "Out: " << counts(after_gamma2, after_out));
}

} // namespace iarc7_vision
