#ifndef _IARC7_VISION_ROOMBA_ESTIMATOR_SETTINGS_HPP_
#define _IARC7_VISION_ROOMBA_ESTIMATOR_SETTINGS_HPP_

namespace iarc7_vision
{

// See vision_node_params.yaml for descriptions
struct RoombaEstimatorSettings {
    double template_pixels_per_meter;
    double roomba_plate_width;
    double roomba_height;

    int ght_pos_thresh;
    int ght_angle_thresh;
    int ght_scale_thresh;
    int ght_canny_low_thresh;
    int ght_canny_high_thresh;
    double ght_dp;
    int ght_levels;
    double ght_angle_step;
    double ght_scale_step;

    int hsv_slice_h_green_min;
    int hsv_slice_h_green_max;
    int hsv_slice_h_red1_min;
    int hsv_slice_h_red1_max;
    int hsv_slice_h_red2_min;
    int hsv_slice_h_red2_max;
    int hsv_slice_s_min;
    int hsv_slice_s_max;
    int hsv_slice_v_min;
    int hsv_slice_v_max;

    int morphology_size;
    int morphology_iterations;

    double bottom_camera_aov;
    bool debug_hsv_slice;
    bool debug_contours;
    bool debug_rects;
    bool debug_detected_rects;
};

} // namespace iarc7_vision

#endif // include guard
