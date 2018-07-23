#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, chain
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf

from sensor_msgs.msg import Image

from iarc7_vision.filterbank import makeRFSfilters
from iarc7_vision.opencv_display_mult_floor import im_show_m

bridge = CvBridge()

def image_callback(data):
    global latest_data
    latest_data = data

radius = 10
kernel_size = radius * 2 + 1
sigmas = [1, 2, 3]
n_sigmas = len(sigmas)
n_orientations = 6

edge, bar, rot = makeRFSfilters(radius=radius,
                                sigmas=sigmas,
                                n_orientations=n_orientations)

all_filters = np.append(edge, bar, axis=0)
all_filters = np.append(all_filters, rot, axis=0)

# Reshape the input vectors into what tensorflow needs
np_filters = np.float32(np.zeros((kernel_size, kernel_size, 1, len(all_filters))))
for x in range(0, kernel_size):
    for y in range(0, kernel_size):
        for z in range(0, len(all_filters)):
            np_filters[x, y, 0, z] = all_filters[z, x, y]

show_filters = False
if show_filters:
    fig, ax = plt.subplots(nrows=n_sigmas*2+1, ncols=n_orientations)
    for x in range(0, len(all_filters)):
        print np.max(np_filters[:, :, 0, x])
        print np.min(np_filters[:, :, 0, x])
        ax[x/n_orientations, x%n_orientations].imshow(np_filters[:, :, 0, x], cmap=plt.cm.gray)
        ax[x/n_orientations, x%n_orientations].set_xticks(())
        ax[x/n_orientations, x%n_orientations].set_yticks(())

tf_filters = tf.constant(np_filters)

placeholder_image = tf.placeholder(tf.float32, shape=(1, 960, 1280, 3))

gray_image = tf.image.rgb_to_grayscale(placeholder_image)

convolved = tf.nn.conv2d(gray_image, tf_filters, strides=[1, kernel_size/3, kernel_size/3, 1], padding='VALID')
squared = tf.square(convolved)
average_size = squared.shape[1]/10
averaged = tf.nn.avg_pool(squared, (1, average_size, average_size, 1), (1, average_size, average_size, 1), 'VALID')

constrasted = tf.image.adjust_contrast(averaged, 1.0)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()

def stretch_contrast(img):
    maximum = np.max(img)
    minimum = np.min(img)

    return (img-minimum)*(1.0/(maximum-minimum))

show_filtered = False
if show_filtered:
    fig, ax = plt.subplots(nrows=n_sigmas*2+1, ncols=n_orientations)
    for x in range(0, len(all_filters)):
        ax[x/n_orientations, x%n_orientations].imshow(result[0, :, :, x], cmap=plt.cm.gray)
        ax[x/n_orientations, x%n_orientations].set_xticks(())
        ax[x/n_orientations, x%n_orientations].set_yticks(())

if show_filtered or show_filters:
    plt.show()
latest_data = None
if __name__ == '__main__':
    rospy.init_node('sharpen_bottom_camera')

    while not rospy.is_shutdown() and rospy.Time.now() == 0:
        pass

    rospy.Subscriber("/bottom_camera/rgb/image_raw", Image, image_callback, queue_size=None)

    while latest_data is None:
        pass

    while not rospy.is_shutdown():
        try:
          image = bridge.imgmsg_to_cv2(latest_data, "rgb8")
        except CvBridgeError as e:
          print(e)

        new_img = np.float32(np.asarray([image]))/255.0

        result = sess.run(constrasted, feed_dict={placeholder_image: new_img})
        print result.shape

        imgs = []
        for i in range(0, len(all_filters)):
            imgs.append(np.uint8(255.0 * stretch_contrast(result[0, :, :, i])))

        # Show all the iamges
        im_show_m(imgs)
        cv2.waitKey(1)
