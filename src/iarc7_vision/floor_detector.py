#!/usr/bin/env python
from timeit import default_timer as timer
import pickle
import glob

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

bridge = CvBridge()


def image_callback(data):
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

    resized_image = cv2.resize(
        cropped, settings.target_size, interpolation=cv2.INTER_LINEAR)

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
    labels = np.asarray(labels)

    # Make sure all the detections aren't of one class
    # before trying to find the boundary line
    if np.sum(labels) > 0 and np.sum(labels) < len(labels):
        # Train an SVM on the spot to find the boundary line
        line_clf = train_boundary_classifier(points, labels)
    else:
        line_clf = None

    if publish_visualization:
        publish_debug(resized_image, points, prediction, line_clf,
                      data.header.stamp)


def publish_debug(resized_image, points, prediction, line_clf, stamp):

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

    if line_clf is not None:
        coefficients = line_clf.coef_
        # Use the coefficients to characterize the line
        # Check for cases where floating point precision will cause all kinds of
        # weird miscalculations
        # Is the line a well defined vertical line?
        if coefficients[0, 0] <= 10**-8 and abs(coefficients[0, 1]) > 10**-8:
            # Find the x intercept
            x_int = int(-line_clf.intercept_[0] / coefficients[0, 1])
            cv2.line(resized_image, (x_int, 0),
                     (x_int, resized_image.shape[0]), (0, 0, 255))
        # Is the line a well defined not vertical line?
        elif abs(coefficients[0, 0]) > 10**-8:
            p1 = (-1, (-line_clf.intercept_[0] / coefficients[0, 0]) +
                  (coefficients[0, 1] / coefficients[0, 0]))
            p2 = (resized_image.shape[1] + 1,
                  (-line_clf.intercept_[0] / coefficients[0, 0]) -
                  ((resized_image.shape[1] + 1) * coefficients[0, 1] /
                   coefficients[0, 0]))
            p1 = (int(p1[0]), int(p1[1]))
            p2 = (int(p2[0]), int(p2[1]))
            cv2.line(resized_image, p1, p2, (0, 0, 255))
        # A well defined line was not found
        else:
            pass

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

    while not rospy.is_shutdown() and rospy.Time.now() == 0:
        pass

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

    image_topic = rospy.get_param('~camera_topic')
    rospy.Subscriber(image_topic, Image, image_callback, queue_size=1)

    rospy.spin()
