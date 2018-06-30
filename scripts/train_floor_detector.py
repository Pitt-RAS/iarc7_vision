#!/usr/bin/env python
from timeit import default_timer as timer

import rosbag
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

def stretch_contrast(img):
    maximum = np.max(img)
    minimum = np.min(img)

    return (img-minimum)*(1.0/(maximum-minimum))

def filter_image_set(images, filters, start_image=0, max_images=None):
    filter_applicator = None
    filtered_images = []
    num_processed = 0
    start_time = timer()
    for topic, msg, t in images.read_messages(topics=['/bottom_camera/height_image']):
        if num_processed >= start_image:
            if filter_applicator is None:
                filter_applicator = ImageFilterApplicator(filters, (msg.image.width, msg.image.height))
            try:
              image = bridge.imgmsg_to_cv2(msg.image, "rgb8")
            except CvBridgeError as e:
              print(e)

            result = filter_applicator.apply_filters(np.asarray([np.float32(image)/255.0]), show_result=False)
            filtered_images.append(result[0, :, :, :])
            #cv2.imshow('T', np.uint8(255.0 * stretch_contrast(result[0, :, :, 0])))
            #cv2.waitKey(1)
        num_processed += 1
        if max_images is not None:
            if num_processed >= max_images + start_image:
                break
    end_time = timer()

    print ('Filtered {} images in {} seconds fps: {}'.format(
           num_processed - start_image,
           end_time-start_time,
           num_processed/(end_time-start_time)))

    return np.asarray(filtered_images)

def get_feature_vectors(images):
    shape = images.shape
    vectors = []
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                vectors.append(images[i, j, k, :4])
    #return vectors
    vectors = np.float64(np.asarray(vectors))
    print vectors.shape
    return vectors

def create_classifier_set(floor_vectors, not_floor_vectors):
    all_vectors = np.append(floor_vectors, not_floor_vectors, axis=0)

    floor_labels = np.full((floor_vectors.shape[0]), 0.0)
    not_floor_labels = np.full((not_floor_vectors.shape[0]), 1.0)
    all_labels = np.append(floor_labels, not_floor_labels, axis=0)

    print all_vectors.shape
    print all_labels.shape
    return all_vectors, all_labels

def train_classifier(vectors, labels):
    clf = SVC(kernel="linear", C=0.025)
    start_time = timer()
    clf.fit(vectors, labels)
    end_time = timer()
    print ('Trained on {} vectors in {} seconds vectors/sec: {}'.format(
           vectors.shape[0],
           end_time-start_time,
           vectors.shape[0]/(end_time-start_time)))
    return clf

if __name__ == '__main__':

    kernel_size = 21
    sigmas = [1, 2, 3]
    n_orientations = 6

    filters = get_RFS_filters_in_tensorflow_format(kernel_size,
                                                   sigmas,
                                                   n_orientations,
                                                   show_filters=False)

    floor_images = rosbag.Bag('sim_floor_set.bag', 'r')
    not_floor_images = rosbag.Bag('sim_gym_set.bag', 'r')

    filtered_floor_images = filter_image_set(floor_images, filters, max_images=100)
    filtered_not_floor_images = filter_image_set(not_floor_images, filters, max_images=100)

    floor_vectors = get_feature_vectors(filtered_floor_images)
    not_floor_vectors = get_feature_vectors(filtered_not_floor_images)
    vectors, labels = create_classifier_set(floor_vectors, not_floor_vectors)

    clf = train_classifier(vectors, labels)

    score = clf.score(vectors, labels)
    print('SCORE OF TRAINED SET: {}'.format(score))

    test_filtered_floor_images = filter_image_set(floor_images, filters, start_image=150, max_images=400)
    test_filtered_not_floor_images = filter_image_set(not_floor_images, filters, start_image=150, max_images=400)
    test_floor_vectors = get_feature_vectors(test_filtered_floor_images)
    test_not_floor_vectors = get_feature_vectors(test_filtered_not_floor_images)
    test_vectors, test_labels = create_classifier_set(test_floor_vectors, test_not_floor_vectors)

    start_time = timer()
    score = clf.score(test_vectors, test_labels)
    end_time = timer()
    print ('Scored {} vectors in {} seconds vectors/sec: {}'.format(
           test_vectors.shape[0],
           end_time-start_time,
           test_vectors.shape[0]/(end_time-start_time)))
    print('SCORE OF TEST SET: {}'.format(score))
