#!/usr/bin/env python  
import rospy
import tf_conversions
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from sensor_msgs.msg import Temperature, Imu


def callBack(msg):
    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "map"
    t.child_frame_id = "box"
    t.transform.rotation.x = msg.pose.orientation.x
    t.transform.rotation.y = msg.pose.orientation.y
    t.transform.rotation.z = msg.pose.orientation.z
    t.transform.rotation.w = msg.pose.orientation.w

    br.sendTransform(t)

if __name__ == '__main__':
      rospy.init_node('tf_broadcaster_imu')
      rospy.Subscriber('/orientation', PoseStamped, callBack)
      rospy.spin()