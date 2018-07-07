#include "iarc7_vision/ColorCorrectionModel.hpp"

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
    cv::cuda::GpuMat in_post_gamma;
    gamma_lut_->transform(in, in_post_gamma, stream);

    std::array<cv::cuda::GpuMat, 3> in_channels;
    cv::cuda::split(in_post_gamma, in_channels.data(), stream);

    std::array<cv::cuda::GpuMat, 3> saturated_masks;
    for (int i = 0; i < 3; i++) {
        cv::cuda::compare(in_channels[i], 255, saturated_masks[i], cv::CMP_GE, stream);
    }

    std::array<cv::cuda::GpuMat, 3> gbr_channels = {{in_channels[1],
                                                     in_channels[2],
                                                     in_channels[0]}};
    cv::cuda::GpuMat gbr_order;
    cv::cuda::merge(gbr_channels.data(), 3, gbr_order, stream);

    std::array<cv::cuda::GpuMat, 3> brg_channels = {{in_channels[2],
                                                     in_channels[0],
                                                     in_channels[1]}};
    cv::cuda::GpuMat brg_order;
    cv::cuda::merge(brg_channels.data(), 3, brg_order, stream);

    cv::cuda::GpuMat out1, out2, out3;
    cv::cuda::multiply(in_post_gamma,
                       cv::Scalar(a00_, a11_, a22_),
                       out1,
                       1.,
                       CV_32FC3,
                       stream);
    cv::cuda::multiply(gbr_order,
                       cv::Scalar(a01_, a12_, a20_),
                       out2,
                       1.,
                       CV_32FC3,
                       stream);
    cv::cuda::multiply(brg_order,
                       cv::Scalar(a02_, a10_, a21_),
                       out3,
                       1.,
                       CV_32FC3,
                       stream);

    cv::cuda::GpuMat out_float1;
    cv::cuda::GpuMat out_float2;
    cv::cuda::add(out1, out2, out_float1, cv::noArray(), CV_32FC3, stream);
    cv::cuda::add(out3,
                  cv::Scalar(offset0_, offset1_, offset2_),
                  out_float2,
                  cv::noArray(),
                  CV_32FC3,
                  stream);

    cv::cuda::GpuMat out_float;
    cv::cuda::add(out_float1, out_float2, out_float, cv::noArray(), CV_32FC3, stream);

    cv::cuda::GpuMat out_before_gamma, out_after_gamma;
    out_float.convertTo(out_before_gamma, CV_8UC3);
    final_lut_->transform(out_before_gamma, out_after_gamma, stream);

    std::array<cv::cuda::GpuMat, 3> out_channels;
    cv::cuda::split(out_after_gamma, out_channels.data(), stream);
    for (int i = 0; i < 3; i++) {
        out_channels[i].setTo(255, saturated_masks[i], stream);
    }
    cv::cuda::merge(out_channels.data(), 3, out, stream);
}

} // namespace iarc7_vision
