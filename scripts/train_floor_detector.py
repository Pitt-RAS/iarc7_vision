#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, chain
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf

from sensor_msgs.msg import Image

from iarc7_vision.filterbank import get_RFS_filters_in_tensorflow_format
from iarc7_vision.image_filter_applicator import ImageFilterApplicator

if __name__ == '__main__':

    kernel_size = 21
    sigmas = [1, 2, 3]
    n_orientations = 6

    filters = get_RFS_filters_in_tensorflow_format(kernel_size,
                                                   sigmas,
                                                   n_orientations,
                                                   show_filters=False)

    filter_applicator = ImageFilterApplicator(filters, (1280, 960))

    img = cv2.imread('/home/levi/sim_view.jpg')
    img = np.asarray([np.float32(img)/255.0])


    filter_applicator.apply_filters(img, show_result=True)
