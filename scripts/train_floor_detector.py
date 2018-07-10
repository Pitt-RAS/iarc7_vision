#!/usr/bin/env python
from timeit import default_timer as timer
import pickle
import glob

import rosbag
import rospkg
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
    num_seen = 0
    for topic, msg, t in images.read_messages(topics=['/bottom_camera/height_image']):
        num_seen += 1
        if num_seen < start_image:
            continue

        if filter_applicator is None:
            target_size = (320, 240)
            filter_applicator = ImageFilterApplicator(filters, target_size)
        try:
          image = bridge.imgmsg_to_cv2(msg.data, "rgb8")
        except CvBridgeError as e:
          print(e)

        height = msg.header.seq/1000.0
        min_height = 0.7
        if height > min_height:
            crop_amount_width  = int(image.shape[1] - min(image.shape[1] / (height / min_height), image.shape[1]))/2
            crop_amount_height = int(image.shape[0] - min(image.shape[0] / (height / min_height), image.shape[0]))/2
            cropped = image[crop_amount_width:image.shape[1]-crop_amount_width,
                            crop_amount_height:image.shape[0]-crop_amount_height]
            resized_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)

            result = filter_applicator.apply_filters(np.asarray([np.float32(resized_image)/255.0]), show_result=False)
            filtered_images.append(result[0, :, :, :])
            #cv2.imshow('T', np.uint8(255.0 * stretch_contrast(result[0, :, :, 0])))
            #cv2.waitKey(1)
            num_processed += 1
        if max_images is not None:
            if num_processed >= max_images:
                break
    end_time = timer()

    print ('Filtered {} images in {} seconds fps: {}'.format(
           num_processed,
           end_time-start_time,
           (num_processed)/(end_time-start_time)))
    print ('Went through {} images'.format(num_seen))

    np_filtered_images = np.asarray(filtered_images)
    print np_filtered_images.shape
    return np_filtered_images, num_seen

def get_feature_vectors(images):
    shape = images.shape
    vectors = []
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                vectors.append(images[i, j, k, :])
    vectors = np.float32(np.asarray(vectors))
    print('Feature vectors shape')
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
    #clf = SVC(kernel="linear", C=25.0)
    clf = SVC(gamma=2, C=2.0)
    start_time = timer()
    clf.fit(vectors, labels)
    end_time = timer()
    print ('Trained on {} vectors in {} seconds vectors/sec: {}'.format(
           vectors.shape[0],
           end_time-start_time,
           vectors.shape[0]/(end_time-start_time)))
    return clf

if __name__ == '__main__':

    kernel_size = 5
    sigmas = [0.2, 0.4, 0.6]
    n_orientations = 6

    filters = get_RFS_filters_in_tensorflow_format(kernel_size,
                                                   sigmas,
                                                   n_orientations,
                                                   show_filters=False)
    filters = filters[:, :, :, :]
    print filters.shape

    rospack = rospkg.RosPack()

    floors     = sorted(glob.glob(rospack.get_path('iarc7_vision') + '/training_bags/drone_bags/floor*'))
    antifloors = sorted(glob.glob(rospack.get_path('iarc7_vision') + '/training_bags/drone_bags/antifloor*'))

    floor_images     = rosbag.Bag(floors[-1], 'r')
    not_floor_images = rosbag.Bag(antifloors[-1], 'r')

    filtered_floor_images, floor_split = filter_image_set(floor_images, filters, start_image=0, max_images=200)
    filtered_not_floor_images, not_floor_split = filter_image_set(not_floor_images, filters, start_image =0, max_images=200)

    floor_vectors = get_feature_vectors(filtered_floor_images)
    not_floor_vectors = get_feature_vectors(filtered_not_floor_images)
    vectors, labels = create_classifier_set(floor_vectors, not_floor_vectors)

    clf = train_classifier(vectors, labels)

    floor_score = clf.score(floor_vectors, np.full((floor_vectors.shape[0]), 0.0))
    not_floor_score = clf.score(not_floor_vectors, np.full((not_floor_vectors.shape[0]), 1.0))
    print('SCORE OF FLOOR TEST SET: {}'.format(floor_score))
    print('SCORE OF NOT FLOOR TEST SET: {}'.format(not_floor_score))

    test_filtered_floor_images, end_floor_image = filter_image_set(floor_images, filters, start_image=floor_split)
    test_filtered_not_floor_images, end_not_floor_image = filter_image_set(not_floor_images, filters, start_image=not_floor_split)
    test_floor_vectors = get_feature_vectors(test_filtered_floor_images)
    test_not_floor_vectors = get_feature_vectors(test_filtered_not_floor_images)

    start_time = timer()
    floor_score = clf.score(test_floor_vectors, np.full((test_floor_vectors.shape[0]), 0.0))
    not_floor_score = clf.score(test_not_floor_vectors, np.full((test_not_floor_vectors.shape[0]), 1.0))

    end_time = timer()
    print ('Scored {} vectors in {} seconds vectors/sec: {}'.format(
           test_floor_vectors.shape[0] + test_not_floor_vectors.shape[0],
           end_time-start_time,
           (test_floor_vectors.shape[0] + test_not_floor_vectors.shape[0])/(end_time-start_time)))
    print('SCORE OF FLOOR TEST SET: {}'.format(floor_score))
    print('SCORE OF NOT FLOOR TEST SET: {}'.format(not_floor_score))

    pickle.dump(clf, open(rospack.get_path('iarc7_vision') + "/clf.params", "wb" ))
