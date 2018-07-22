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
    def __init__(self, filterbank, incoming_resolution, stride, average_size):
        self.filterbank = filterbank
        self.incoming_resolution = incoming_resolution
        self.average_size = average_size
        self.stride = stride

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
                                          self.stride,
                                          self.stride,
                                          1],
                                padding='VALID')

        squared = tf.square(convolved)

        print('Pre average size: {}'.format(squared.shape))
        self.averaged = tf.nn.avg_pool(squared,
                                       (1, self.average_size, self.average_size, 1),
                                       (1, self.average_size, self.average_size, 1),
                                       'VALID')
        print('Post average size: {}'.format(self.averaged.shape))

        # Average red
        test = 3 * self.average_size
        self.average_rgb = tf.nn.avg_pool(self.placeholder_image,
                                         (1, test, test, 1),
                                         (1, test, test, 1),
                                          padding='VALID')

        if output_graph:
            writer = tf.summary.FileWriter('.')
            writer.add_graph(tf.get_default_graph())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def apply_filters(self, image, show_result=False):

        result = self.sess.run([self.averaged,
                                self.average_rgb],
                                feed_dict={self.placeholder_image: image})
        #result = np.asarray(result)
        #result = result.flatten(axis=0)
        #print result[0].shape
        #print result[1].shape
        all_results = np.append(result[0], result[1], axis=3)
        #print all_results.shape

        if show_result:
            imgs = []
            for i in range(0, all_results.shape[3]):
                imgs.append(np.uint8(255.0 * all_results[0, :, :, i]))
            im_show_m(imgs)
            cv2.waitKey()
        return all_results
