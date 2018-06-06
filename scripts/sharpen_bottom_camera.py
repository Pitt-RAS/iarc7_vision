#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

from opencv_display_mult import im_show_m

bridge = CvBridge()

def image_callback(data):

    try:
      image = bridge.imgmsg_to_cv2(data, "rgb8")
    except CvBridgeError as e:
      print(e)

    # Images to display
    imgs = []

    # Get the image image width and height
    ih, iw, channels = image.shape

    # Convert to HSV and extract V
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    imgs.append(v)

    # Run a canny on the v channel
    high = 15000
    ratio = 3
    edges = cv2.Canny(v,high/ratio,high,apertureSize = 7)
    imgs.append(edges)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.dilate(edges,kernel,iterations = 1)
    imgs.append(erosion)

    mask_applied = cv2.bitwise_and(image, image, mask=erosion)
    #cv2.imshow('mask applied', mask_applied)

    sharpened_msg = bridge.cv2_to_imgmsg(mask_applied, encoding="rgb8")
    sharpened_msg.header.stamp = rospy.Time.now()
    sharpened_pub.publish(sharpened_msg)

    # Show all the iamges
    #im_show_m(imgs)
    #cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('sharpen_bottom_camera')

    while not rospy.is_shutdown() and rospy.Time.now() == 0:
        pass

    sharpened_pub = rospy.Publisher("/bottom_camera/sharpened", Image, queue_size=3)

    rospy.Subscriber("/bottom_camera/rgb/image_raw", Image, image_callback)


    rospy.spin()