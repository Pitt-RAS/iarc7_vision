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

from sensor_msgs.msg import Image

from iarc7_vision.filterbank import get_RFS_filters_in_tensorflow_format
from iarc7_vision.image_filter_applicator import ImageFilterApplicator

bridge = CvBridge()

def stretch_contrast(img):
    maximum = np.max(img)
    minimum = np.min(img)

    return (img-minimum)*(1.0/(maximum-minimum))

def filter_image_set(images, filters, max_images=None):
    filter_applicator = None
    filtered_images = []
    num_processed = 0
    start_time = timer()
    for topic, msg, t in images.read_messages(topics=['/bottom_camera/height_image']):
        if filter_applicator is None:
            filter_applicator = ImageFilterApplicator(filters, (msg.image.width, msg.image.height))
        try:
          image = bridge.imgmsg_to_cv2(msg.image, "rgb8")
        except CvBridgeError as e:
          print(e)

        result = filter_applicator.apply_filters(np.asarray([np.float32(image)/255.0]), show_result=False)
        filtered_images.append(result)
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
           num_processed/(end_time-start_time)))

    return filtered_images

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

    filter_image_set(floor_images, filters, max_images=200)
    filter_image_set(not_floor_images, filters, max_images=200)
