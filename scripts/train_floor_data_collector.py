#!/usr/bin/env python

import rospy
import tf2_ros

from sensor_msgs.msg import Image
from iarc7_msgs.msg import HeightImage

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

def image_callback(data):
    # Lookup the height from the tf tree
    try:
        trans = tf_buffer.lookup_transform(
                    'map',
                    'bottom_camera_rgb_optical_frame',
                    data.header.stamp,
                    rospy.Duration(0.01))
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as ex:
        msg = "Train floor data collector: Exception when looking up transform"
        rospy.logerr("Transform error: {}".format(msg))
        rospy.logerr(ex.message)

    data.header.seq = int(trans.transform.translation.z*1000)
    pub.publish(data)

if __name__ == '__main__':
    rospy.init_node('train_floor_data_collector')

    while not rospy.is_shutdown() and rospy.Time.now() == 0:
        pass

    pub = rospy.Publisher("/bottom_camera/height_image", Image, queue_size=3)

    rospy.Subscriber("/corrected_image", Image, image_callback)

    rospy.spin()
