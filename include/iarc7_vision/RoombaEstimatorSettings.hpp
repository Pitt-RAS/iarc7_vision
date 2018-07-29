#ifndef _IARC7_VISION_ROOMBA_ESTIMATOR_SETTINGS_HPP_
#define _IARC7_VISION_ROOMBA_ESTIMATOR_SETTINGS_HPP_

namespace iarc7_vision
{

// See vision_node_params.yaml for descriptions
struct RoombaEstimatorSettings {
    double roomba_plate_width;
    double roomba_plate_height;
    double roomba_height;

    int detection_image_width;

    int min_roomba_blob_size;
    int max_roomba_blob_size;

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

    double max_relative_error;

    double uncertainty_scale;

    double bottom_camera_aov;
    bool debug_hsv_slice;
    bool debug_contours;
    bool debug_detected_rects;
};

} // namespace iarc7_vision

#endif // include guard
