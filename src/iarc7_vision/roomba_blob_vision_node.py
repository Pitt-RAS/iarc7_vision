#!/usr/bin/env python
"""
Vision node to detect roombas based of red or green colored pixels, which are
distinct from their black & white peppered background.

Algorithm:

    1. Perform HSV color filtering to select only the relevant red and green
       pixels.
    2. Use median filtering to remove stray pixels while retaining the edge
       details on the red and green pixel clumps.
    3. Find contours on the remaining pixel clumps.
    4. Calculate the bounding box of any pixel clumps larger than 10x10 pixels.
    5. Translate the camera coordinates to map frame using CameraInfo and the
       transform library

Preconditions:

    - Simulator: `bottom_camera_resolution` is enabled
    - Simulator: `ground_truth_roombas` is disabled

Execution:

    $ rosrun iarc7_vision roomba_blob_vision_node.py

.. note::
    Contrary to online code, the do_transform_vector3 method DOES NOT create a
    deep copy of the transform first, and thus this method will set the
    translation of the transform to the origin! This can be seen in the file:
    `python2.7/dist-packages/tf2_geometry_msgs/tf2_geometry_msgs.py`

.. note:
    The default behavior is to act as a camera cv node with debugging
    turned off.

:docformat: reStructuredText
"""
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Point, TransformStamped, Vector3Stamped
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from iarc7_msgs.msg import OdometryArray
from nav_msgs.msg import Odometry

import tf2_ros
import tf2_geometry_msgs
import image_geometry

import numpy as np
import math
import copy

NODE_NAME = "roomba_blob_vision_node"
# From: iarc7_simulator/sim/src/sim/builder/robots/Roomba.py
ROOMBA_HEIGHT = 0.065

class Debugger(object):
    
    def __init__(self, image_on=True, rviz_on=True):
        self._image_enabled = image_on
        self._rviz_enabled = rviz_on
        if self._rviz_enabled:
            self.rviz_pub = rospy.Publisher('visualization_marker', Marker, \
                                                  queue_size=10)
            self.rviz_frame = "/map"
            self.rviz_namespace = NODE_NAME
        
    def image_grid(self, *args):
        """
        Displays up to 4 images in a single window

        :param *args: Up to 4 numpy.ndarray images
        :return: None

        .. note::
            All image arguments must have the same number of color channels,
            same dimensions, and same datatype. This is not validated.
        """
        if not self._image_enabled or len(args)==0:
            return
        f1 = args[0]
        black = np.zeros(f1.shape, dtype=f1.dtype)
        f2 = black if len(args) < 2 else args[1]
        f3 = black if len(args) < 3 else args[2]
        f4 = black if len(args) < 4 else args[3]
        row1 = np.hstack((f1, f2))
        row2 = np.hstack((f3, f4))
        columns = np.vstack((row1,row2))
        cv2.imshow('Image Debugger', columns)
        cv2.waitKey(1)

    def rviz_lines(self, points, line_id):
        """
        Displays a line_list in rviz with the given points and id

        :param points: list of points composing line segments
        :type points: builtin list
        :return: None
        """
        if not self._rviz_enabled:
            return
        line = Marker()
        line.header.frame_id = self.rviz_frame
        line.header.stamp = rospy.Time.now()
        line.ns = self.rviz_namespace
        line.action = Marker.ADD
        line.id = line_id
        line.type = Marker.LINE_LIST

        line.pose.orientation.w = 1.0
        line.scale.x = 0.03
        line.color.r = 1.0
        line.color.a = 1.0

        line.points = points

        self.rviz_pub.publish(line)

class ImageRoombaFinder(object):
    
    def __init__(self):
        self.debug = Debugger(False, False)

    def filter_ranges(self, frame, ranges):
        """
        Filter the image to only areas between the HSV values described in
        ranges. This will not alter the original image.
    
        :param frame: 3-channel image to apply HSV range filters to
        :type frame: numpy.ndarray
        :param ranges: list of 3-element tuples
        :return: 3-channel filtered image
        :rtype: numpy.ndarray
    
        .. note::
            In OpenCV, the HSV ranges are [0,180], [0,255], [0,255].
        """
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create black image the same size as frame
        out = np.zeros(frame.shape, dtype=frame.dtype)
        for r in ranges:
            # Create mask
            mask = cv2.inRange(hsv_image, r[0], r[1])
            # Apply the mask
            out = cv2.add(out, cv2.bitwise_and(frame, frame, mask=mask))
        return out

    def filter_roombas(self, frame):
        """
        Keep only the roombas in the frame
        
        :param frame: raw BGR image in which to find roombas
        :type frame: numpy.ndarray of shape 3 x WIDTH x HEIGHT with dtype uint8
        :return: filtered frame
        :rtype: numpy.ndarray of shape 3 x WIDTH x HEIGHT with dtype uint8
    
        .. note::
            From a few sample images, it seems that the saturation can go as
            low as 13%, which is equivalent to 33/255. Images are white when
            saturation is 0. 50/255 seems reasonable for minimum value. When
            the value is 0, the image is black.
    
        .. note::
            The numbers used in the ranges need to be altered slightly
            depending on the brightness of the `frame` being processed. The
            upper limits don't need to be changed, but the saturation and value
            constants for the lower limits will need to change if the image is
            significantly darker or lighter.
        """
        # TODO This will be tuned for varying brightness levels
        ranges = np.array([
            [[ 30,100, 20], [ 90,255,255]], # Green
            [[  0,100, 20], [  8,255,255]], # Low Red
            [[165,100, 20], [179,255,255]], # High Red
            # [[ 30, 25, 50], [ 90,255,255]], # Green
            # [[  0, 50, 50], [  8,255,255]], # Low Red
            # [[165, 80,100], [179,255,255]], # High Red
        ])
        # frame = cv2.GaussianBlur(frame,(5,5),0)
        filtered_frame = self.filter_ranges(frame, ranges)
        filtered_frame = cv2.medianBlur(filtered_frame, 5)
        return filtered_frame

    def bound_roombas(self, frame):
        """
        Find bounding boxes for all pixel clumps (roombas) with dimensions
        larger than 10 pixels on each side. Also draws the bounding rects onto
        the original frame.
    
        :param img_gray: single channel image used to find edges
        :type img_gray: numpy.ndarray of shape 1 x WIDTH x HEIGHT & dtype uint8
        :param img: original three channel (BGR) image to draw boxes on
        :type img: numpy.ndarray of shape 3 x WIDTH x HEIGHT and dtype uint8
        :return: Iterator object containing (x, y) pairs
        """
        # remove everything from the original image that is not a roomba
        roombas = self.filter_roombas(frame)
        # convert original image to single channel
        frame_gray = cv2.cvtColor(roombas, cv2.COLOR_BGR2GRAY)
        # find the contours in this single channel image
        # RETR_EXTERNAL won't match boxes inside other boxes
        contours, _ = cv2.findContours(frame_gray,cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rect = cv2.boundingRect(c)
            # Skip tiny boxes
            # TODO change minimum size based on drone position
            if rect[2] < 10 or rect[3] < 10: continue
            x, y, w, h = rect # unpack the rect
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 4)
            yield (x + w/2, y + h/2)
        # After the generator is done, debug the image
        self.debug.image_grid(frame, roombas)

class CameraProcessor(ImageRoombaFinder):

    def __init__(self, debugging_on):
        rospy.init_node(NODE_NAME, anonymous=True)

        self.debug = Debugger(debugging_on, debugging_on)

        self.roombas = []

        self.bridge = CvBridge()
        self.camera = image_geometry.PinholeCameraModel()

        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)
        # Make sure at least one transform exists before starting
        self.tf_buffer.lookup_transform('map', 'bottom_camera_optical',
                                        rospy.Time(0), rospy.Duration(3.0))

        rospy.Subscriber("/bottom_image_raw/image", Image, self.callback)
        rospy.Subscriber("/bottom_image_raw/camera_info", CameraInfo,
                         self.camera.fromCameraInfo)
        self.publisher = rospy.Publisher("/roombas", OdometryArray,
                                         queue_size=10)

        # Starup the loop
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")
        
    def callback(self, data):
        """
        Receive the callback from the camera, process the image, and publish
        the roomba locations found

        :param data: Image
        :type data: sensor_msgs.msg.Image
        :return: None
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = cv2.resize(cv_image,None,fx=0.5,fy=0.5, 
            #                       interpolation=cv2.INTER_CUBIC)
            trans = self.tf_buffer.lookup_transform('map',
                                'bottom_camera_optical', rospy.Time(0))
            self.filter_frame(cv_image, trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, \
                tf2_ros.ExtrapolationException, CvBridgeError) as e:
            rospy.logerr(e)


    def filter_frame(self, frame, trans):
        """
        Applies a four-step algorithm to find all the roombas in a given BGR
        image.
    
        :param frame: raw BGR image in which to find roombas
        :type frame: numpy.ndarray of shape 3 x WIDTH x HEIGHT with dtype uint8
        :return: None
    
        .. note::
            Does not return data, only debugs the result
    
        .. note::
            The image origin is in the upper-left hand corner
        """
        points = []
    
        for img_coords in self.bound_roombas(frame):
            cam_ray = Vector3Stamped()
            cam_ray.vector.x, cam_ray.vector.y, cam_ray.vector.z = \
                                    self.camera.projectPixelTo3dRay(img_coords)
            
            # Convert that camera ray to world space (map frame)
            # See note in script header about do_transform_vector3
            map_ray = tf2_geometry_msgs.do_transform_vector3(cam_ray,
                                                          copy.deepcopy(trans))
            
            # Scale the direction to hit the ground (plane z=0)
            direction_scale = (trans.transform.translation.z - ROOMBA_HEIGHT) \
                              / map_ray.vector.z
            roomba_pos = Point()
            roomba_pos.x = trans.transform.translation.x - \
                           map_ray.vector.x * direction_scale
            roomba_pos.y = trans.transform.translation.y - \
                           map_ray.vector.y * direction_scale
            roomba_pos.z = 0
    
            # Debug the roomba line
            points.append(trans.transform.translation)
            points.append(roomba_pos)

            # Add the roomba to array and publish
            # TODO This can be improved
            sq_tolerance = 0.1 if len(self.roombas) < 10 else 1000
            index = -1
            for i in xrange(len(self.roombas)):
                if (roomba_pos.x - self.roombas[i].x)**2 + \
                   (roomba_pos.y - self.roombas[i].y)**2 < sq_tolerance:
                    self.roombas[i] = roomba_pos
                    index = i
                    break
            if index == -1:
                index = len(self.roombas)
                self.roombas.append(roomba_pos)

            out_msg = OdometryArray()
            for pt in self.roombas:
                item = Odometry()
                item.header.frame_id = "/map"
                item.pose.pose.position = pt
                out_msg.data.append(item) 
            self.publisher.publish(out_msg)

                
        self.debug.rviz_lines(points, 0)

class VideoProcessor(ImageRoombaFinder):
    """
    This class can be used to test the CV on any video. This way, it is
    easy to test already captured video that may have different lighting or
    distortion than the camera captures from the sim.
    """

    def __init__(self, file_path):
        """
        :param file_path: path to video file
        """
        super(VideoProcessor, self).__init__()
        self.debug = Debugger(True, False)
        cap = cv2.VideoCapture(file_path)
        while True:
            retval, frame = cap.read()
            if not retval: # Exit if there is not a frame
                break
            for _ in self.bound_roombas(frame):
                pass
            cv2.waitKey(1)

if __name__ == '__main__':
    # Uncomment the following line to run on a sample video, and comment out
    # the CameraProcessor line below.
    # VideoProcessor("../Experiments/green_test.mp4")

    # Run the main node functionality.
    # Change False to True to turn on debugging
    CameraProcessor(False)

    # Make sure all the cv windows closed
    cv2.destroyAllWindows()
