#!/usr/bin/env python

import rospy
import tf2_ros

from sensor_msgs.msg import Image

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

    data.header.frame_id = str(trans.transform.translation.z)
    pub.publish(data)

if __name__ == '__main__':
    rospy.init_node('train_floor_data_collector')

    while not rospy.is_shutdown() and rospy.Time.now() == 0:
        pass

    pub = rospy.Publisher("/bottom_camera/height_image", Image, queue_size=3)

    image_topic = rospy.get_param('~camera_topic')
    rospy.Subscriber(image_topic, Image, image_callback)

    rospy.spin()
