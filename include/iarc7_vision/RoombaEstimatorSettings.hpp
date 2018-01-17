#ifndef _IARC7_VISION_ROOMBA_ESTIMATOR_SETTINGS_HPP_
#define _IARC7_VISION_ROOMBA_ESTIMATOR_SETTINGS_HPP_

namespace iarc7_vision
{

// See vision_node_params.yaml for descriptions
struct RoombaEstimatorSettings {
    double pixels_per_meter;
    double roomba_plate_width;
    double roomba_height;
    int ght_levels;
    int ght_dp;
    int ght_votes_threshold;
    int camera_canny_threshold;
    int template_canny_threshold;
    double bottom_camera_aov;
    bool debug_hsv_slice;
    bool debug_contours;
    bool debug_rects;
};

} // namespace iarc7_vision

#endif // include guard
