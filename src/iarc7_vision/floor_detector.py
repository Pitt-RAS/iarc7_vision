#!/usr/bin/env python
from timeit import default_timer as timer
import pickle

import rospy
import rospkg
import tf2_ros
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from itertools import product, chain
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
                    'map',
                    'bottom_camera_rgb_optical_frame',
                    data.header.stamp,
                    rospy.Duration(0.01))
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as ex:
        msg = "Train floor data collector: Exception when looking up transform"
        rospy.logerr("Transform error: {}".format(msg))
        rospy.logerr(ex.message)

    start_time = timer()
    height = trans.transform.translation.z

    # TODO min height parameter
    min_height = 0.7
    crop_amount_width  = int(image.shape[1] - min(image.shape[1] / (height / min_height), image.shape[1]))/2
    crop_amount_height = int(image.shape[0] - min(image.shape[0] / (height / min_height), image.shape[0]))/2
    cropped = image[crop_amount_width:image.shape[1]-crop_amount_width,
                    crop_amount_height:image.shape[0]-crop_amount_height]
    resized_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)

    result = filter_applicator.apply_filters(np.asarray([np.float32(resized_image)/255.0]), show_result=False)
    end_time = timer()
    rospy.logdebug('Filtered 1 image in {} seconds fps: {}'.format(
                   end_time-start_time,
                   1/(end_time-start_time)))

    shape = result.shape
    vectors = []
    for j in range(0, shape[1]):
        for k in range(0, shape[2]):
            vectors.append(result[0, j, k, :])

    #return vectors
    vectors = np.float32(np.asarray(vectors))

    prediction = clf.predict(vectors)

    prediction = np.reshape(prediction, (shape[1], shape[2]))

    # Convert prediction to list of points
    # Points are centered in prediction boxes
    points = []
    labels = []
    height_pix_offset = (1.0 / prediction.shape[0] / 2.0) * resized_image.shape[0]
    width_pix_offset  = (1.0 / prediction.shape[1] / 2.0) * resized_image.shape[1]

    for i in range(0, prediction.shape[0]):
        for j in range(0, prediction.shape[1]):
            new_point = (((float(i) / prediction.shape[0]) * resized_image.shape[0]) + height_pix_offset,
                         ((float(j) / prediction.shape[1]) * resized_image.shape[1]) + width_pix_offset)
            points.append(new_point)
            labels.append(prediction[i, j])
    points = np.asarray(points)
    labels = np.asarray(labels)

    block_height = resized_image.shape[0] / prediction.shape[0]
    block_width = resized_image.shape[1] / prediction.shape[1]
    resized_image = resized_image / 2
    for i in range(0, prediction.shape[0]):
        for j in range(0, prediction.shape[1]):
            if prediction[i, j] == 0:
                resized_image[i*block_height:(i+1)*block_height,
                              j*block_width:(j+1)*block_width,
                              1] = 200
            elif prediction[i, j] == 1:
                resized_image[i*block_height:(i+1)*block_height,
                              j*block_width:(j+1)*block_width,
                              0] = 200
    for p in points:
        resized_image[int(p[0]), int(p[1]), :] = 0

    sum_labels = np.sum(labels)
    if sum_labels != 0 and sum_labels < len(labels):
        clf2 = train_classifier(points, labels)

        # If we have a vertical line
        if clf2.coef_[0, 0] <= 10 ** -8 and abs(clf2.coef_[0, 1]) > 10 ** -8:
            # Find the x intercept
            x_int = int(-clf2.intercept_[0] / clf2.coef_[0, 1])
            cv2.line(resized_image, (x_int, 0), (x_int, resized_image.shape[0]), (0, 0, 255))

        # There is not a vertical line
        elif abs(clf2.coef_[0, 0]) > 10 ** -8:
            p1 = (-1, (-clf2.intercept_[0] / clf2.coef_[0, 0]) + (clf2.coef_[0, 1]/clf2.coef_[0, 0]))
            p2 = (resized_image.shape[1]+1, (-clf2.intercept_[0] / clf2.coef_[0, 0]) - ((resized_image.shape[1]+1) * clf2.coef_[0, 1]/clf2.coef_[0, 0]))
            p1 = (int(p1[0]), int(p1[1]))
            p2 = (int(p2[0]), int(p2[1]))
            cv2.line(resized_image, p1, p2, (0, 0, 255))
        else:
            # The estimator probably returned all 0 coefficients and nothing can be done
            pass

    debug_msg = bridge.cv2_to_imgmsg(resized_image, encoding="rgb8")
    debug_msg.header.stamp = data.header.stamp
    detections_pub.publish(debug_msg)

def train_classifier(vectors, labels):
    clf = SVC(kernel="linear", C=0.025)
    start_time = timer()
    clf.fit(vectors, labels)
    end_time = timer()
    rospy.logdebug ('Trained on {} vectors in {} seconds vectors/sec: {}'.format(
                    vectors.shape[0],
                    end_time-start_time,
                    vectors.shape[0]/(end_time-start_time)))
    return clf

if __name__ == '__main__':
    rospy.init_node('floor_detector')

    while not rospy.is_shutdown() and rospy.Time.now() == 0:
        pass

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    kernel_size = 5
    sigmas = [0.2, 0.4, 0.6]
    n_orientations = 6

    filters = get_RFS_filters_in_tensorflow_format(kernel_size,
                                                   sigmas,
                                                   n_orientations,
                                                   show_filters=False)
    filters = filters[:, :, :, :]

    target_size = (320, 240)
    filter_applicator = ImageFilterApplicator(filters, target_size)

    rospack = rospkg.RosPack()

    clf = pickle.load(open( rospack.get_path('iarc7_vision') + "/clf.params", "rb" ))

    #clf.decision_function(((0, None)))

    rospy.Subscriber("/bottom_camera/rgb/image_raw", Image, image_callback)

    detections_pub = rospy.Publisher("/floor_detector/detections_image", Image, queue_size=3)

    rospy.spin()
