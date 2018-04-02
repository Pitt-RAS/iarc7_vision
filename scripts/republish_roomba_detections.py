#!/usr/bin/env python

'''
Publishes roomba detections as markers for visualization in RViz
Camera frame is published as a polygon, roomba detections are published as spheres

Camera frame topic: /roomba_detection_frame_marker
Roomba observations topic: /roomba_detection_markers
'''

import rospy
from geometry_msgs.msg import Point, PolygonStamped
from iarc7_msgs.msg import RoombaDetectionFrame
from visualization_msgs.msg import Marker

def callback(msg):
    frame = PolygonStamped()
    frame.header = msg.header
    frame.polygon = msg.detection_region
    frame_pub.publish(frame)

    marker = Marker()
    marker.header = msg.header
    marker.ns = 'roomba_observations'
    marker.id = 0
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.MODIFY

    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    marker.color.r = 0
    marker.color.g = 1
    marker.color.b = 0
    marker.color.a = 1

    marker.lifetime = rospy.Duration(0.5)
    marker.frame_locked = False

    for roomba in msg.roombas:
        point = Point()
        point.x = roomba.pose.x
        point.y = roomba.pose.y
        point.z = 0
        marker.points.append(point)

    vis_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('republish_roomba_detections')
    frame_pub = rospy.Publisher('/roomba_detection_frame_marker',
                                PolygonStamped,
                                queue_size=5)
    vis_pub = rospy.Publisher('/roomba_detection_markers',
                              Marker,
                              queue_size=5)
    rospy.Subscriber('/detected_roombas', RoombaDetectionFrame, callback)
    rospy.spin()
