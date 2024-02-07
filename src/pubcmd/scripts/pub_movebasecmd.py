#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import rospy
import roslib
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from px4_command.msg import command_acc
import numpy as np

import tf
import math


class CMD:

    def __init__(self):
        rospy.init_node('pub_movebasecmd',anonymous=True)
        self.pub_cmd = rospy.Publisher("/px4/command",command_acc,queue_size=1)
        self.sub_cmd = rospy.Subscriber("/cmd_vel",Twist,self.cmd_cb)
        rospy.spin()
    
    def cmd_cb(self,msg):
        data = command_acc()
        #data.header = msg.header
        data.command = 1
        data.sub_mode = 0
        data.comid = 0

        #data.pos_sp[0] = msg.position.x
        #data.pos_sp[1] = msg.position.y
        #data.pos_sp[2] = msg.position.z

        data.vel_sp[0] = msg.linear.x
        data.vel_sp[1] = msg.linear.y
        #data.vel_sp[2] = msg.velocity.z

        #data.acc_sp[0] = msg.acceleration.x
        #data.acc_sp[1] = msg.acceleration.y
        #data.acc_sp[2] = msg.acceleration.z
     
        #data.yaw_sp = msg.yaw*180/math.pi
        data.yaw_rate_sp = msg.angular.z*180/math.pi
   
        self.pub_cmd.publish(data)

if __name__ == '__main__':
   try:
      cmd = CMD()
   except rospy.ROSInterruptException:
      pass
