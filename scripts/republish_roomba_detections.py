#!/usr/bin/env python

'''
Publishes roomba detections as markers for visualization in RViz
Camera frame is published as a polygon, roomba detections are published as spheres

Camera frame topic: /roomba_detection_frame_marker
Roomba observations topic: /roomba_detection_markers
'''

import math
import rospy
from geometry_msgs.msg import Point, PolygonStamped
from iarc7_msgs.msg import RoombaDetectionFrame
from visualization_msgs.msg import Marker, MarkerArray

count = 0

def callback(msg):
    frame = PolygonStamped()
    frame.header = msg.header
    frame.polygon = msg.detection_region
    frame_pub.publish(frame)

    marker_arr = MarkerArray()
    for roomba in msg.roombas:
        marker = Marker()
        marker.header = msg.header
        marker.ns = 'roomba_observations'

        global count
        marker.id = count
        count += 1

        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.scale.x = 0.05
        marker.scale.y = roomba.box_uncertainty
        marker.scale.z = 0.0

        if roomba.flip_certainty > 0.5:
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 1
            marker.color.a = 1
        else:
            marker.color.r = 1
            marker.color.g = 1
            marker.color.b = 0
            marker.color.a = 1

        marker.lifetime = rospy.Duration(0.5)
        marker.frame_locked = False

        point = Point()
        point.x = roomba.pose.x
        point.y = roomba.pose.y
        point.z = 0
        marker.points.append(point)

        point = Point()
        point.x = roomba.pose.x + 0.2*math.cos(roomba.pose.theta)
        point.y = roomba.pose.y + 0.2*math.sin(roomba.pose.theta)
        point.z = 0
        marker.points.append(point)

        marker_arr.markers.append(marker)

    vis_pub.publish(marker_arr)

if __name__ == '__main__':
    rospy.init_node('republish_roomba_detections')
    frame_pub = rospy.Publisher('/roomba_detection_frame_marker',
                                PolygonStamped,
                                queue_size=5)
    vis_pub = rospy.Publisher('/roomba_detection_markers',
                              MarkerArray,
                              queue_size=5)
    rospy.Subscriber('/detected_roombas', RoombaDetectionFrame, callback)
    rospy.spin()
