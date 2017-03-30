#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros

from sensor_msgs.msg import Image

from opencv_display_mult import im_show_m

bridge = CvBridge()
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

def get_height():
    try:
        trans = tf_buffer.lookup_transform(
                    'map',
                    'quad',
                    rospy.Time.now()-rospy.Duration(0.11),
                    rospy.Duration(0.01))
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as ex:
        msg = "Exception when looking up transform to check that we're grounded"
        rospy.logerr("Takeofftask: {}".format(msg))
        rospy.logerr(ex.message)
        return
    return trans.transform.translation.z

def image_callback(data):

    try:
      image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    height = get_height()

    # Images to display
    imgs = []

    # Get the image image width and height
    ih, iw, channels = image.shape

    # m/px = (height - camera z offset) * tan(HFOV/2) / (image width pixels / 2)
    # tan VFOV/2 = 0.5/(1.27-0.2) Measured using sim
    meters_per_px = (height - 0.2) * (0.5/(1.27-0.2))/(iw/2)

    # desired m_px used to keep kernel sizes relative to our features
    desired_meters_per_px = 1.0/150.0
    scale_factor = meters_per_px / desired_meters_per_px

    image = cv2.resize(image, (int(iw*scale_factor), int(ih*scale_factor)), interpolation = cv2.INTER_LINEAR)
    ih, iw, channels = image.shape
    print(image.shape)
    imgs.append(image)

    # Convert to HSV and extract V
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    imgs.append(v)

    # Run a canny on the v channel
    high = 35000
    ratio = 2.5
    edges = cv2.Canny(v,high/ratio,high,apertureSize = 7)
    imgs.append(edges)
    
    # Now run line detector, require the lines be a percent of the image height
    # The percentage determines how far out of the arena borders we can travel
    hough_weight = int(ih * 0.30)
    lines = cv2.HoughLines(edges,1,np.pi/360,hough_weight)

    # Draw the lines for visualization of the output
    drawn_on = np.copy(image)
    drawn_on2 = np.zeros((ih,iw,3), np.uint8)
    draw_lines(drawn_on, lines)
    draw_lines(drawn_on2, lines)

    imgs.append(drawn_on)
    imgs.append(drawn_on2)

    # Show all the iamges
    im_show_m(imgs)
    cv2.waitKey(1)

def draw_lines(img, lines):
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 3000*(-b))
            y1 = int(y0 + 3000*(a))
            x2 = int(x0 - 3000*(-b))
            y2 = int(y0 - 3000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

if __name__ == '__main__':
    rospy.init_node('ground_velocity_estimator')

    while not rospy.is_shutdown() and rospy.Time.now() == 0:
        pass
    start_time = rospy.Time.now()

    rospy.Subscriber("/bottom_image_raw/image", Image, image_callback)
    rospy.spin()
