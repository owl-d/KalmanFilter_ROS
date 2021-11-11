#! /usr/bin/env python

import rospy
from sensor_msgs.msg import Imu, MagneticField, Temperature
from geometry_msgs.msg import Vector3, PoseStamped, Twist, TransformStamped
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply
import numpy as np
import numpy.linalg as lin
import math
import sys
import rospy
import tf_conversions
import tf2_ros

class Algorithm(object):

    def __init__(self):
        self._imu_sub = rospy.Subscriber("mavros/imu/data_raw", Imu, self.CallBack)
        self._mag_sub = rospy.Subscriber("mavros/imu/mag", MagneticField, self.MagCallBack)
        self.pose_pub = rospy.Publisher("pose", PoseStamped, queue_size = 10)
        self.velocity_pub = rospy.Publisher("velocity", Twist, queue_size = 10)
        self._pose = PoseStamped()
        self._velocity = Twist()
        self._gyro = Vector3()
        self._acc = Vector3()
        self._mag = Vector3()

        self.body_v_x = 0
        self.body_v_y = 0 #velocity
        self.local_v_x = 0
        self.local_v_y = 0
        self.local_p_x = 0
        self.local_p_y = 0 #position
        
        self._dt = 0.02
        self._H = np.zeros((3,4))
        self._Q = 1e-6*np.eye(4)
        self._R = 2*np.eye(3)
        self._V = np.eye(3)
        self._P = 0.001*np.eye(4)
        self._A = np.zeros((4,4))
        self._K = np.zeros((4,4))
        self._x = np.array([[1],[0],[0],[0]]) #quarternion, angular position : output
        self._xp = np.zeros((4,1))
        self._Pp = np.zeros((4,4))
        self._z1 = np.zeros((3,1))
        self._z2 = np.zeros((3,1))
        self._h = np.zeros((3,1))

    def CallBack(self, msg):
        I = np.eye(4)
        self._gyro.x = msg.angular_velocity.x
        self._gyro.y = msg.angular_velocity.y
        self._gyro.z = msg.angular_velocity.z

        self._acc.x = msg.linear_acceleration.x
        self._acc.y = msg.linear_acceleration.y
        self._acc.z = msg.linear_acceleration.z

        #Priori System Estimate : gyro
        self.create_A()
        self._xp = self._A.dot(self._x)
        self._Pp = self._A.dot(self._P.dot(self._A.T)) + self._Q

        #Correction Stage1 : with acc
        self.create_H1()
        self.GetKalmanGain()
        qe1 = self._K.dot(self._z1 - self._h)
        qe1[3][0] = 0
        q1 = self._xp + qe1
        temp = I - self._K.dot(self._H)
        P1 = temp.dot(self._Pp)

        #Correction Stage2 : with mag
        self.create_H2()
        self.GetKalmanGain()
        qe2 = self._K.dot(self._z2 - self._h)
        qe2[1][0] = 0
        qe2[2][0] = 0
        self._x = q1 + qe2
        temp = I - self._K.dot(self._H)
        self._P = temp.dot(P1)

        #output(normalization)
        Q_abs = math.sqrt(self._x[0][0]**2 + self._x[1][0]**2 + self._x[2][0]**2 + self._x[3][0]**2)
        self._x = self._x/Q_abs
        self.getposition()

        self._pose.pose.position.x = self.local_p_x
        self._pose.pose.position.y = self.local_p_y
        self._pose.pose.position.z = 0
        self._pose.pose.orientation.w = self._x[0][0]
        self._pose.pose.orientation.x = self._x[1][0]
        self._pose.pose.orientation.y = self._x[2][0]
        self._pose.pose.orientation.z = self._x[3][0]
        self._pose.header.stamp = rospy.Time.now()
        self._pose.header.frame_id = "map"
        self.pose_pub.publish(self._pose)

        self._velocity.linear.x = self.body_v_x
        self._velocity.linear.y = self.body_v_y
        self._velocity.linear.z = 0
        self._velocity.angular.x = self._gyro.x
        self._velocity.angular.y = self._gyro.y
        self._velocity.angular.z = self._gyro.z
        self.velocity_pub.publish(self._velocity)


    def MagCallBack(self, msg):
        self._mag.x = msg.magnetic_field.x
        self._mag.y = msg.magnetic_field.y
        self._mag.z = msg.magnetic_field.z

    def getposition(self):
        quat = [self._x[0][0], self._x[1][0], self._x[2][0], self._x[3][0]]
        _, _, yaw = euler_from_quaternion(quat)

        local_acc_x = math.cos(yaw)*self._acc.x - math.sin(yaw)*self._acc.y
        local_acc_y = math.sin(yaw)*self._acc.x + math.cos(yaw)*self._acc.y
        self.body_v_x += self._acc.x*self._dt
        self.body_v_y += self._acc.y*self._dt #velocity
        self.local_v_x += local_acc_x*self._dt
        self.local_v_y += local_acc_y*self._dt
        self.local_p_x += self.local_v_x*self._dt/2
        self.local_p_y += self.local_v_y*self._dt/2 #position
        
        # self.local_p_x += local_acc_x*self._dt*self._dt/2
        # self.local_p_y += local_acc_y*self._dt*self._dt/2

    def create_A(self) :
        #A
        ohm = np.zeros((4,4))
        ohm[0][1] = -self._gyro.x
        ohm[0][2] = -self._gyro.y
        ohm[0][3] = -self._gyro.z
        ohm[1][0] = self._gyro.x
        ohm[1][2] = self._gyro.z
        ohm[1][3] = -self._gyro.y
        ohm[2][0] = self._gyro.y
        ohm[2][1] = -self._gyro.z
        ohm[2][3] = self._gyro.x
        ohm[3][0] = self._gyro.z
        ohm[3][1] = self._gyro.y
        ohm[3][2] = -self._gyro.x
        I = np.eye(4)
        self._A = I + self._dt/2*ohm

    def create_H1(self):
        #H_k1
        self._H[0][0] = -2*self._x[2][0]
        self._H[0][1] = 2*self._x[3][0]
        self._H[0][2] = -2*self._x[0][0]
        self._H[0][3] = 2*self._x[1][0]
        self._H[1][0] = 2*self._x[1][0]
        self._H[1][1] = 2*self._x[0][0]
        self._H[1][2] = 2*self._x[3][0]
        self._H[1][3] = 2*self._x[2][0]
        self._H[2][0] = 2*self._x[0][0]
        self._H[2][1] = -2*self._x[1][0]
        self._H[2][2] = -2*self._x[2][0]
        self._H[2][3] = 2*self._x[3][0]

        #h_1(qp)
        self._h[0][0] = 2*self._x[1][0]*self._x[3][0] - 2*self._x[0][0]*self._x[2][0]
        self._h[1][0] = 2*self._x[0][0]*self._x[1][0] + 2*self._x[2][0]*self._x[3][0]
        self._h[2][0] = self._x[0][0]**2 - self._x[1][0]**2 - self._x[2][0]**2 + self._x[3][0]**2
        self._h = 9.81 * self._h

        #z1
        self._z1 = [[self._acc.x], [self._acc.y], [self._acc.z]]

        #R1
        self._R = 2*np.eye(3)

    def create_H2(self):
        #H_k2
        self._H[0][0] = 2*self._x[3][0]
        self._H[0][1] = 2*self._x[2][0]
        self._H[0][2] = 2*self._x[1][0]
        self._H[0][3] = 2*self._x[0][0]
        self._H[1][0] = 2*self._x[0][0]
        self._H[1][1] = -2*self._x[1][0]
        self._H[1][2] = -2*self._x[2][0]
        self._H[1][3] = -2*self._x[3][0]
        self._H[2][0] = -2*self._x[1][0]
        self._H[2][1] = -2*self._x[0][0]
        self._H[2][2] = 2*self._x[3][0]
        self._H[2][3] = 2*self._x[2][0]

        #h_2(qp)
        self._h[0][0] = 2*self._x[1][0]*self._x[2][0] + 2*self._x[0][0]*self._x[3][0]
        self._h[1][0] = self._x[0][0]**2 - self._x[1][0]**2 - self._x[2][0]**2 - self._x[3][0]**2
        self._h[2][0] = 2*self._x[2][0]*self._x[3][0] - 2*self._x[0][0]*self._x[1][0]

        #z2
        self._z2 = [[self._mag.x], [self._mag.y], [self._mag.z]]

        #R2
        self._R = np.eye(3)

    def GetKalmanGain(self) :
        temp = self._H.dot(self._Pp.dot(self._H.T)) + self._V.dot(self._R.dot(self._V.T))
        inverse_temp = lin.inv(temp)
        self._K = self._Pp.dot(self._H.T.dot(inverse_temp))

if __name__ == '__main__' :
    rospy.init_node("kalman_IMU")
    Algorithm()
    rospy.spin()