#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import rospy
import roslib
import sys
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand
from px4_command.msg import command_acc
import numpy as np
import geometry_msgs
#import airsim

import tf
import math

def to_quaternion(pitch, roll, yaw):
    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)

    #q = Quaternionr()
    q = geometry_msgs.msg.Pose()
    q.orientation.w = t0 * t2 * t4 + t1 * t3 * t5 #w
    q.orientation.x = t0 * t3 * t4 - t1 * t2 * t5 #x
    q.orientation.y = t0 * t2 * t5 + t1 * t3 * t4 #y
    q.orientation.z = t1 * t2 * t4 - t0 * t3 * t5 #z
    return q

class CMD:

    def __init__(self):
        rospy.init_node('pub_cmd',anonymous=True)
        self.pub_pose = rospy.Publisher("/2dgoal",PoseStamped,queue_size=1)
        self.pub_cmd = rospy.Publisher("/px4/command",command_acc,queue_size=1)
        self.sub_cmd = rospy.Subscriber("/planning/pos_cmd",PositionCommand,self.cmd_cb)
        rospy.spin()

    def cmd_cb(self,msg):
        data = PoseStamped()
        data.header = msg.header

        data.pose.position.x = msg.position.x
        data.pose.position.y = msg.position.y
        data.pose.position.z = msg.position.z

        q = to_quaternion(0,0,msg.yaw)
        data.pose.orientation.x = q.orientation.x
        data.pose.orientation.y = q.orientation.y
        data.pose.orientation.z = q.orientation.z
        data.pose.orientation.w = q.orientation.w
   
        self.pub_pose.publish(data)

        data = command_acc()
        data.header = msg.header
        data.command = 0
        data.sub_mode = 0
        data.comid = 0

        data.pos_sp[0] = msg.position.x
        data.pos_sp[1] = msg.position.y
        data.pos_sp[2] = msg.position.z

        data.vel_sp[0] = msg.velocity.x
        data.vel_sp[1] = msg.velocity.y
        data.vel_sp[2] = msg.velocity.z

        data.acc_sp[0] = msg.acceleration.x
        data.acc_sp[1] = msg.acceleration.y
        data.acc_sp[2] = msg.acceleration.z
     
        data.yaw_sp = msg.yaw*180/math.pi
        data.yaw_rate_sp = msg.yaw_dot*180/math.pi
   
        self.pub_cmd.publish(data)



if __name__ == '__main__':
   try:
      cmd = CMD()
   except rospy.ROSInterruptException:
      pass
