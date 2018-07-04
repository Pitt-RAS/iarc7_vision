#include "iarc7_vision/ColorCorrectionModel.hpp"

#include "ros_utils/ParamUtils.hpp"

namespace iarc7_vision {

ColorCorrectionModel::ColorCorrectionModel(const ros::NodeHandle& nh)
    : r_a1_(ros_utils::ParamUtils::getParam<double>(nh, "r_a1")),
      r_b1_(ros_utils::ParamUtils::getParam<double>(nh, "r_b1")),
      r_a2_(ros_utils::ParamUtils::getParam<double>(nh, "r_a2")),
      r_b2_(ros_utils::ParamUtils::getParam<double>(nh, "r_b2")),
      g_a1_(ros_utils::ParamUtils::getParam<double>(nh, "g_a1")),
      g_b1_(ros_utils::ParamUtils::getParam<double>(nh, "g_b1")),
      g_a2_(ros_utils::ParamUtils::getParam<double>(nh, "g_a2")),
      g_b2_(ros_utils::ParamUtils::getParam<double>(nh, "g_b2")),
      b_a1_(ros_utils::ParamUtils::getParam<double>(nh, "b_a1")),
      b_b1_(ros_utils::ParamUtils::getParam<double>(nh, "b_b1")),
      b_a2_(ros_utils::ParamUtils::getParam<double>(nh, "b_a2")),
      b_b2_(ros_utils::ParamUtils::getParam<double>(nh, "b_b2")),
      gamma_(ros_utils::ParamUtils::getParam<double>(nh, "gamma"))
{
    cv::Mat lut(cv::Size(256, 1), CV_8UC3);
    for (size_t i = 0; i < 256; i++) {
        uint8_t r_result;
        uint8_t g_result;
        uint8_t b_result;
        {
            const double inner = (r_a1_*i + r_b1_) / 255.;
            const double after_gamma = std::pow(inner, gamma_) * 255.;
            const double result = r_a2_*after_gamma + r_b2_;
            r_result = result < 0
                     ? 0
                     : (result > 255 ? 255 : static_cast<char>(result));
        }
        {
            const double inner = (g_a1_*i + g_b1_) / 255.;
            const double afteg_gamma = std::pow(inner, gamma_) * 255.;
            const double result = g_a2_*afteg_gamma + g_b2_;
            g_result = result < 0
                     ? 0
                     : (result > 255 ? 255 : static_cast<char>(result));
        }
        {
            const double inner = (b_a1_*i + b_b1_) / 255.;
            const double afteb_gamma = std::pow(inner, gamma_) * 255.;
            const double result = b_a2_*afteb_gamma + b_b2_;
            b_result = result < 0
                     ? 0
                     : (result > 255 ? 255 : static_cast<char>(result));
        }
        lut.at<cv::Vec3b>(0, i) = cv::Vec3b(r_result, g_result, b_result);
    }

    lut_ = cv::cuda::createLookUpTable(lut);
}

void ColorCorrectionModel::correct(const cv::cuda::GpuMat& in,
                                   cv::cuda::GpuMat& out,
                                   cv::cuda::Stream& stream) const
{
    lut_->transform(in, out, stream);
}

} // namespace iarc7_vision
