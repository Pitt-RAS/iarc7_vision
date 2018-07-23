#!/usr/bin/env python
from timeit import default_timer as timer
import pickle
import glob
import tf2_geometry_msgs
from tf2_geometry_msgs import Vector3Stamped, PointStamped
from geometry_msgs.msg import Point
import math
from tf import transformations
from collections import deque

import rospy
import rospkg
import tf2_ros
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from sklearn.svm import SVC

from sensor_msgs.msg import Image

from iarc7_vision.filterbank import get_RFS_filters_in_tensorflow_format
from iarc7_vision.image_filter_applicator import ImageFilterApplicator

from visualization_msgs.msg import Marker

from iarc7_msgs.msg import Boundary

bridge = CvBridge()


class DebugData(object):
    def __init__(self):
        self.resized_image = None
        self.points = None
        self.prediction = None
        self.line_clf = None
        self.center_point = None
        self.failed_arena_edge = False

def image_callback(data):
    debug_data = DebugData()
    detection = find_boundary_line(data, debug_data)

    if detection is None:
        detection = [0, 0, 0, 0, 0, data.header.stamp]

    filter_detections(detection)

    if publish_visualization:
        publish_debug(debug_data, data.header.stamp)

def filter_detections(detection):
    global detections
    detections.append(detection)

    start_time = detection[5]

    detections = [x for x in detections if x[5] > start_time - max_detection_lag]

    if len(detections) > max_detections_queued:
        detections.pop(0)
    #print len(detections), min_boundary_detections
    if(len(detections) >= min_boundary_detections):
        # Sort all the detections
        left_detections = []
        right_detections = []
        top_detections = []
        bottom_detections = []

        for d in detections:
            if d[0] > 0:
                left_detections.append(np.asarray((d[0], d[4])))
            elif d[1] > 0:
                right_detections.append(np.asarray((d[1], d[4])))
            elif d[2] > 0:
                top_detections.append(np.asarray((d[2], d[4])))
            elif d[3] > 0:
                bottom_detections.append(np.asarray((d[3], d[4])))

        #print len(left_detections), len(right_detections), len(top_detections), len(bottom_detections)

        publish_detection(validate_detection(left_detections), Boundary.left, detection[5])
        publish_detection(validate_detection(right_detections), Boundary.right, detection[5])
        publish_detection(validate_detection(top_detections), Boundary.top, detection[5])
        publish_detection(validate_detection(bottom_detections), Boundary.bottom, detection[5])

def publish_detection(distance, b_type, stamp):
    if distance is not None:
        msg = Boundary()
        msg.header.stamp = stamp
        msg.boundary_type = b_type
        msg.position = distance
        detection_publisher.publish(msg)

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Check if vertical wall
        LINE_LENGTH = 100
        if b_type == Boundary.left or b_type == Boundary.right:
            m_p1 = Point(-LINE_LENGTH, distance, 0)
            m_p2 = Point(LINE_LENGTH, distance, 0)
        else:
            m_p1 = Point(distance, -LINE_LENGTH, 0)
            m_p2 = Point(distance, LINE_LENGTH, 0)
        marker.points = [m_p1, m_p2]
        final_line_pos_marker_pub.publish(marker)

def validate_detection(detection):
    if(len(detection) < min_boundary_detections):
        #print('Not enough detections {} {}'.format(len(detection), min_boundary_detections))
        return

    distances = np.asarray(detection)[:, 1]
    if np.var(distances) > max_boundary_variance:
        return
    #print('Variance {}'.format(np.var(distances)))

    return np.average(distances)

def find_boundary_line(data, debug_data):
    try:
        image = bridge.imgmsg_to_cv2(data, "rgb8")
    except CvBridgeError as e:
        rospy.logerr(e)

    # Lookup the height from the tf tree
    try:
        trans = tf_buffer.lookup_transform(
            'map', 'bottom_camera_rgb_optical_frame', data.header.stamp,
            rospy.Duration(0.10))
        height = trans.transform.translation.z
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as ex:
        msg = "Train floor data collector: Exception when looking up transform"
        rospy.logerr("Transform error: {}".format(msg))
        rospy.logerr(ex.message)
        return

    if height < settings.min_height:
        return

    start_time = timer()

    crop_amount_width = int(image.shape[1] - min(image.shape[1] / (
        height / settings.min_height), image.shape[1])) / 2
    crop_amount_height = int(image.shape[0] - min(image.shape[0] / (
        height / settings.min_height), image.shape[0])) / 2
    cropped = image[crop_amount_height:image.shape[0] - crop_amount_height,
                    crop_amount_width:image.shape[1] - crop_amount_width]

    focal_length = math.sqrt(image.shape[1]**2 + image.shape[0]**2) / (2 * math.tan(afov / 2))
    new_afov = 2 * math.atan(math.sqrt(cropped.shape[1]**2 + cropped.shape[0]**2) / focal_length)
    resize_fractions = np.asarray((float(settings.target_size[0]) / cropped.shape[1], float(settings.target_size[1]) / cropped.shape[0]))

    resized_image = cv2.resize(
        cropped, settings.target_size, interpolation=cv2.INTER_LINEAR)
    debug_data.resized_image = resized_image

    result = filter_applicator.apply_filters(
        np.asarray([np.float32(resized_image) / 255.0]), show_result=False)

    end_time = timer()
    rospy.logdebug('Filtered 1 image in {} seconds fps: {}'.format(
        end_time - start_time, 1 / (end_time - start_time)))

    shape = result.shape
    vectors = []
    for j in range(0, shape[1]):
        for k in range(0, shape[2]):
            vectors.append(result[0, j, k, :])

    vectors = np.asarray(vectors)

    prediction = settings.clf.predict(vectors)

    prediction = np.reshape(prediction, (shape[1], shape[2]))
    debug_data.prediction = prediction

    # Convert prediction values to positions with the labels
    # based on the known location of the avaraging squares
    block_height = (settings.average_size - 1) * settings.stride + 1
    block_width = block_height

    # Points are centered in prediction boxes
    points = []
    labels = []
    for i in range(0, prediction.shape[0]):
        for j in range(0, prediction.shape[1]):
            new_point = ((block_height / 2) + (i * block_height) +
                         (settings.kernel_size / 2) +
                         (i * (settings.kernel_size / 2)), (block_width / 2) +
                         (j * block_width) + (settings.kernel_size / 2) +
                         (j * (settings.kernel_size / 2)))
            points.append(new_point)
            labels.append(prediction[i, j])
    points = np.asarray(points)
    debug_data.points = points

    labels = np.asarray(labels)

    # Make sure all the detections aren't of one class
    # before trying to find the boundary line
    if np.sum(labels) > 0 and np.sum(labels) < len(labels):
        # Train an SVM on the spot to find the boundary line
        line_clf = train_boundary_classifier(points, labels)
    else:
        line_clf = None

    debug_data.line_clf = line_clf
    if line_clf is None:
        return

    # Get a list of points that are on the antifloor side of the line
    point_classifications = line_clf.predict(points)

    if point_classifications.size - np.sum(point_classifications) < min_floor_patches:
        debug_data.failed_arena_edge = True
        return

    floor_side_floor = (np.ones(point_classifications.shape) - point_classifications) * (np.ones(point_classifications.shape) - labels)

    ratio = np.sum(floor_side_floor) / (point_classifications.shape[0] - np.sum(point_classifications))
    if ratio < min_floor_appearance_ratio:
        debug_data.failed_arena_edge = True
        return

    if np.sum(point_classifications) < min_anti_floor_patches:
        debug_data.failed_arena_edge = True
        return

    anti_floor_side_anti_floor = point_classifications * labels

    ratio = np.sum(anti_floor_side_anti_floor) / np.sum(point_classifications)
    if ratio < min_anti_floor_appearance_ratio:
        debug_data.failed_arena_edge = True
        return

    edge_mask = np.pad(np.zeros((shape[1]-2, shape[2]-2)), 1, 'constant', constant_values = 1).flatten()

    anti_floor_side_anti_floor_edge = edge_mask * anti_floor_side_anti_floor
    if np.sum(anti_floor_side_anti_floor_edge) < min_anti_floor_on_edge:
        debug_data.failed_arena_edge = True
        return

    # Set up the equations of the hyperplane equation
    # c1*x + c2*y + d = 0
    c1 = line_clf.coef_[0, 1]
    c2 = line_clf.coef_[0, 0]
    d = line_clf.intercept_[0]
    #rospy.logerr('d: {}'.format(d))

    # Test for the equation of the line that will be used
    # (1) y = -(c1/c2)x - (d/c2) if abs(c1) < abs(img_height * c2)
    # (2) x = -(d/c1) if abs(c1) >= abs(img_height * c2) (assumes that c2/c1 ~= 0)

    # If using equation (1)
    if abs(c1) < abs(c2 * resized_image.shape[0]):
        # Test all four sides to find the intersection points
        #rospy.logerr('c1: {} c2: {}'.format(c1, c2))
        points = []
        a = -(c1/c2)
        b = -(d/c2)

        # Test intersection with top of viewport
        top_intersect = -(b/a)
        #rospy.logerr('b: {} a: {}'.format(b, a))
        #rospy.logerr('top_intersect {}'.format(top_intersect))
        if top_intersect >= 0 and top_intersect <= resized_image.shape[1]:
            points.append((top_intersect, 0.0))

        # Test intersection with bottom of viewport
        bottom_intersect = (resized_image.shape[0] - b) / a
        #rospy.logerr('bottom_intersect {}'.format(bottom_intersect))
        if bottom_intersect >= 0 and bottom_intersect <= resized_image.shape[1]:
            points.append((bottom_intersect, float(resized_image.shape[0])))

        # Test intersection left side of viewport
        left_intersect = b
        #rospy.logerr('left_intersect {}'.format(left_intersect))
        if left_intersect >= 0 and left_intersect <= resized_image.shape[0]:
            points.append((0.0, left_intersect))

        # Test intersection with right side of viewport
        right_intersect = a * resized_image.shape[1] + b
        #rospy.logerr('right_intersect {}'.format(right_intersect))
        if right_intersect >= 0 and right_intersect <= resized_image.shape[0]:
            points.append((float(resized_image.shape[1]), right_intersect))

        # If there are 0 points then the line didn't intersect with the viewport
        if len(points) == 0:
            return

        # If there are not two points at this point there was an error
        if len(points) != 2:
            rospy.logerr('Floor detector error calculating intersection points with equation 1')
            return

        p1 = points[0]
        p2 = points[1]

    # If using equation (2)
    else:
        x_intercept = -d/c1
        if x_intercept < 0 or x_intercept > resized_image.shape[1]:
            return
        p1 = (x_intercept, 0.0)
        p2 = (x_intercept -(c2/c1)*resized_image.shape[0], float(resized_image.shape[0]))

    center_point = (p2[0] + (p1[0] - p2[0])/2.0, p2[1] + (p1[1] - p2[1])/2.0)
    debug_data.center_point = center_point

    # Find a vector that describes the orientation of the line in the optical frame
    direction = np.asarray(p1) - np.asarray(p2)
    direction = direction / np.linalg.norm(direction)

    try:
        heading_trans = tf_buffer.lookup_transform(
            'level_quad', 'heading_quad', data.header.stamp,
            rospy.Duration(0.10))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as ex:
        msg = "Train floor data collector: Exception when looking up transform for heading"
        rospy.logerr("Transform error: {}".format(msg))
        rospy.logerr(ex.message)
        return

    # Flip over the vector
    flipped_over_vector = np.asarray([-direction[1], -direction[0]])

    # Rotate the vector into the heading quad frame
    rotation_matrix = np.asarray([[math.cos(-camera_rotation), -math.sin(-camera_rotation)],
                                  [math.sin(-camera_rotation), math.cos(-camera_rotation)]])

    heading_frame_vector = np.matmul(flipped_over_vector, rotation_matrix)

    direction_vector = Vector3Stamped()
    direction_vector.vector.x = heading_frame_vector[0]
    direction_vector.vector.y = heading_frame_vector[1]

    direction_quad = tf2_geometry_msgs.do_transform_vector3(direction_vector, heading_trans)

    marker = Marker()
    marker.header.frame_id = "level_quad"
    marker.header.stamp = data.header.stamp
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0

    rot = math.atan(direction_quad.vector.y / direction_quad.vector.x)
    quat = transformations.quaternion_about_axis(rot, [0, 0, 1])

    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker.scale.x = 1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0 # Don't forget to set the alpha!
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker_pub.publish(marker)

    # Get the position of the wall using the camera transform stuff
    undistorted_pix_pos = center_point / resize_fractions
    # Get the ray
    pix_ray = [undistorted_pix_pos[0] - cropped.shape[1] / 2, undistorted_pix_pos[1] - cropped.shape[0] / 2, focal_length]
    unit_pix_ray = pix_ray / np.linalg.norm(pix_ray)

    camera_ray = Vector3Stamped()
    camera_ray.vector.x = unit_pix_ray[0]
    camera_ray.vector.y = unit_pix_ray[1]
    camera_ray.vector.z = unit_pix_ray[2]

    map_ray = tf2_geometry_msgs.do_transform_vector3(camera_ray, trans)

    ray_scale = abs(trans.transform.translation.z / map_ray.vector.z)

    # Map ray is now a vector that's orientated in the level_quad frame but it's
    # origin is at the bottom_camera frame
    map_ray.vector.x *= ray_scale
    map_ray.vector.y *= ray_scale
    map_ray.vector.z *= ray_scale

    try:
        camera_trans = tf_buffer.lookup_transform(
            'bottom_camera_rgb_optical_frame', 'level_quad', data.header.stamp,
            rospy.Duration(0.10))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as ex:
        msg = "Train floor data collector: Exception when looking up transform for vector transformation"
        rospy.logerr("Transform error: {}".format(msg))
        rospy.logerr(ex.message)
        return

    # Find the final line position
    wall_pos = PointStamped()
    wall_pos.point.x = map_ray.vector.x + camera_trans.transform.translation.x
    wall_pos.point.y = map_ray.vector.y + camera_trans.transform.translation.y
    wall_pos.point.z = map_ray.vector.z + camera_trans.transform.translation.z

    # Find the distance from the dividing line based on a point that will not be on the line
    # Figure out which test point of the detections grid to use
    # Assume the camera has discrete 90 degree rotations
    if camera_rotation == 0:
        test_point_coordinate = np.asarray((0, 0))
    elif camera_rotation == -1.5707:
        test_point_coordinate = np.asarray((resized_image.shape[1], 0))
    else:
        rospy.logerr('FLOOR DETECTOR CAN NOT USE CAMERA ROTATION')

    line_side = line_clf.decision_function((test_point_coordinate, ))

    marker = Marker()
    marker.header.frame_id = 'level_quad'
    marker.header.stamp = data.header.stamp
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0 # Don't forget to set the alpha!
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    # Transform the detection point into the map frame
    try:
        drone_trans = tf_buffer.lookup_transform(
            'map', 'level_quad', data.header.stamp,
            rospy.Duration(0.10))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as ex:
        msg = "Train floor data collector: Exception when looking up transform for vector transformation"
        rospy.logerr("Transform error: {}".format(msg))
        rospy.logerr(ex.message)
        return

    map_wall_pos = tf2_geometry_msgs.do_transform_point(wall_pos, drone_trans)

    # Check if vertical wall
    LINE_LENGTH = 1
    if math.pi / 4 >= rot and rot >= -math.pi / 4:
        # If left
        if np.sign(line_side) == 1:
            detection = [1, 0, 0, 0, map_wall_pos.point.y, data.header.stamp]
            rospy.logwarn('LEFT WALL')
        else:
            detection = np.asarray([0, 1, 0, 0, map_wall_pos.point.y, data.header.stamp])
            rospy.logwarn('RIGHT WALL')
        m_p1 = Point(-LINE_LENGTH, wall_pos.point.y, wall_pos.point.z)
        m_p2 = Point(LINE_LENGTH, wall_pos.point.y, wall_pos.point.z)
        marker.points = [m_p1, m_p2]
    # if horizontal wall
    elif rot > math.pi / 4 or -math.pi / 4 > rot:
        # If top
        if np.sign(line_side) == 1:
            detection = [0, 0, 1, 0, map_wall_pos.point.x, data.header.stamp]
            rospy.logwarn('TOP WALL')
        else:
            detection = [0, 0, 0, 1, map_wall_pos.point.x, data.header.stamp]
            rospy.logwarn('BOTTOM WALL')
        m_p1 = Point(wall_pos.point.x, -LINE_LENGTH, wall_pos.point.z)
        m_p2 = Point(wall_pos.point.x, LINE_LENGTH, wall_pos.point.z)
        marker.points = [m_p1, m_p2]

    line_pos_marker_pub.publish(marker)
    return detection

def publish_debug(data, stamp):
    resized_image = data.resized_image
    points = data.points
    prediction = data.prediction
    line_clf = data.line_clf
    center_point = data.center_point

    if resized_image is None or points is None or prediction is None:
        return

    block_height = (settings.average_size - 1) * settings.stride + 1
    block_width = block_height

    resized_image = resized_image / 2
    for i in range(0, prediction.shape[0]):
        for j in range(0, prediction.shape[1]):
            if prediction[i, j] == 0:
                resized_image[
                    (i * block_height) + (settings.kernel_size / 2) +
                    (i * (settings.kernel_size / 2)):(i * block_height) +
                    (settings.kernel_size / 2) +
                    (i * (settings.kernel_size / 2)) + block_height,
                    (j * block_width) + (settings.kernel_size / 2) +
                    (j * (settings.kernel_size / 2)):(j * block_width) +
                    (settings.kernel_size / 2) +
                    (j * (settings.kernel_size / 2)) + block_width, 1] = 200
            elif prediction[i, j] == 1:
                resized_image[
                    (i * block_height) + (settings.kernel_size / 2) +
                    (i * (settings.kernel_size / 2)):(i * block_height) +
                    (settings.kernel_size / 2) +
                    (i * (settings.kernel_size / 2)) + block_height,
                    (j * block_width) + (settings.kernel_size / 2) +
                    (j * (settings.kernel_size / 2)):(j * block_width) +
                    (settings.kernel_size / 2) +
                    (j * (settings.kernel_size / 2)) + block_width, 0] = 200
    for p in points:
        resized_image[int(p[0]), int(p[1]), :] = 0

    # Try to draw the line classifiers line, sometimes impossible due to
    # precision problems
    try:
        if line_clf is not None:
            coefficients = line_clf.coef_
            # Use the coefficients to characterize the line
            # Check for cases where floating point precision will cause all kinds of
            # weird miscalculations
            c1 = line_clf.coef_[0, 1]
            c2 = line_clf.coef_[0, 0]
            d = line_clf.intercept_[0]
            # Is the line a well defined vertical line?
            if abs(c1) < abs(c2 * resized_image.shape[0]):
                p1 = (0, -d / c2)
                p2 = (resized_image.shape[1],
                      (-d / c2) - (resized_image.shape[1] * c1 / c2))
                p1 = (int(p1[0]), int(p1[1]))
                p2 = (int(p2[0]), int(p2[1]))
                cv2.line(resized_image, p1, p2, (0, 0, 255))
            else:
                # Find the x intercept
                x_int = int(-d / c1)
                if x_int >= 0 and x_int <= resized_image.shape[1]:
                    cv2.line(resized_image,
                             (x_int, 0),
                             (x_int, resized_image.shape[0]),
                             (0, 0, 255))
    except ZeroDivisionError as e:
        rospy.logerr('Caught ZeroDivisionError in floor detector')
        return

    if data.failed_arena_edge:
        cv2.putText(resized_image, 'No edge', (2, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))

    if center_point is not None:
        resized_image[int(center_point[1]), int(center_point[0]), 0] = 255
        cv2.putText(resized_image, 'c: {0:.2f} {0:.2f}'.format(center_point[0], center_point[1]), (2, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))

    debug_msg = bridge.cv2_to_imgmsg(resized_image, encoding="rgb8")
    debug_msg.header.stamp = stamp
    debug_visualization_pub.publish(debug_msg)


def train_boundary_classifier(vectors, labels):
    clf = SVC(kernel="linear", C=0.025)
    start_time = timer()
    clf.fit(vectors, labels)
    end_time = timer()
    rospy.logdebug(
        'Trained on {} vectors in {} seconds vectors/sec: {}'.format(
            vectors.shape[0], end_time - start_time,
            vectors.shape[0] / (end_time - start_time)))
    return clf


def load_classifier():
    rospack = rospkg.RosPack()

    postfix = rospy.get_param('~classifier_settings_postfix')
    revision = rospy.get_param('~revision_name')

    if revision == 'latest':
        classifiers = sorted(glob.glob(rospack.get_path('iarc7_vision') \
                                       + '/classifiers/floor_classifier_params_r*_' \
                                       + postfix \
                                       + '.clf'))
        filename = classifiers[-1]
    else:
        filename = rospack.get_path('iarc7_vision') \
                   + '/classifiers/floor_classifier_params_r' \
                   + str(revision) \
                   + '.clf'

    rospy.loginfo('Floor detector settings file: {}'.format(filename))
    clf = pickle.load(open(filename, "rb"))
    return clf


class SettingsObject(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    rospy.init_node('floor_detector')

    settings = load_classifier()

    min_floor_patches = rospy.get_param('~min_floor_patches')
    min_floor_appearance_ratio = rospy.get_param('~min_floor_appearance_ratio')
    min_anti_floor_patches = rospy.get_param('~min_anti_floor_patches')
    min_anti_floor_appearance_ratio = rospy.get_param('~min_anti_floor_appearance_ratio')
    min_anti_floor_on_edge = rospy.get_param('~min_anti_floor_on_edge')

    camera_rotation = rospy.get_param('~camera_rotation')
    afov = rospy.get_param('~afov')

    detections = deque()
    min_boundary_detections = rospy.get_param('~min_boundary_detections')
    max_detections_queued = rospy.get_param('~max_detections_queued')
    max_detection_lag = rospy.Duration(rospy.get_param('~max_detection_lag'))
    max_boundary_variance = rospy.get_param('~max_boundary_std_dev')**2

    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and rospy.Time.now() == 0:
        rate.sleep()

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    filters = get_RFS_filters_in_tensorflow_format(
        settings.kernel_size,
        settings.sigmas,
        settings.n_orientations,
        show_filters=False)

    filter_applicator = ImageFilterApplicator(
        filters, settings.target_size, settings.stride, settings.average_size)

    publish_visualization = rospy.get_param('~publish_visualization')

    if publish_visualization:
        debug_visualization_pub = rospy.Publisher(
            '/floor_detector/detections_image', Image, queue_size=1)

    marker_pub = rospy.Publisher('/floor_detector/line_direction', Marker, queue_size=1)
    line_pos_marker_pub = rospy.Publisher('/floor_detector/line_position', Marker, queue_size=1)
    final_line_pos_marker_pub = rospy.Publisher('/floor_detector/boundaries_marker', Marker, queue_size=1)

    detection_publisher = rospy.Publisher('/floor_detector/boundaries', Boundary, queue_size=10)

    image_topic = rospy.get_param('~camera_topic')
    rospy.Subscriber(image_topic, Image, image_callback, queue_size=1)

    rospy.spin()
