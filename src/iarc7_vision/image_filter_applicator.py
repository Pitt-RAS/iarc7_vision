#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from opencv_display_mult_floor import im_show_m

def stretch_contrast(img):
    maximum = np.max(img)
    minimum = np.min(img)

    return (img-minimum)*(1.0/(maximum-minimum))

class ImageFilterApplicator:
    def __init__(self, filterbank, incoming_resolution):
        self.filterbank = filterbank
        self.incoming_resolution = incoming_resolution

        self._construct_tensor_graph()

    def _construct_tensor_graph(self, output_graph=False):
        self.placeholder_image = tf.placeholder(tf.float32,
                                                shape=(
                                                  1,
                                                  self.incoming_resolution[1],
                                                  self.incoming_resolution[0],
                                                  3))

        gray_image = tf.image.rgb_to_grayscale(self.placeholder_image)

        filterbank_shape = self.filterbank.shape
        tf_filters = tf.constant(self.filterbank)
        convolved = tf.nn.conv2d(gray_image,
                                 tf_filters,
                                 strides=[1,
                                          3,
                                          3,
                                          1],
                                padding='SAME')

        squared = tf.square(convolved)

        # TODO square shapes need to change with height
        print('Pre average size: {}'.format(squared.shape))
        average_size = 20 #squared.shape[1]/2
        self.averaged = tf.nn.avg_pool(squared,
                                       (1, average_size, average_size, 1),
                                       (1, average_size, average_size, 1),
                                       'VALID')

        if output_graph:
            writer = tf.summary.FileWriter('.')
            writer.add_graph(tf.get_default_graph())

        self.sess = tf.Session()

    def apply_filters(self, image, show_result=False):

        result = self.sess.run(self.averaged, feed_dict={self.placeholder_image: image})

        if show_result:
            imgs = []
            for i in range(0, self.filterbank.shape[3]):
                imgs.append(np.uint8(255.0 * stretch_contrast(result[0, :, :, i])))
            im_show_m(imgs)
            cv2.waitKey(1)
        return result
