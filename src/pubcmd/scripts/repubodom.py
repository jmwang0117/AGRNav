#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import rospy
import roslib
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand
from geometry_msgs.msg import Twist
import numpy as np

import tf
import math
#import airsim


class CMD:

    def __init__(self):
        rospy.init_node('repubodom',anonymous=True)
        self.pub_cmd = rospy.Publisher("/odom_h",Odometry,queue_size=1)
        self.odom_cmd = rospy.Subscriber("/odom", Odometry, self.odom_cb)


        rospy.spin()
    
    
    def odom_cb(self, msg):
        reodom = msg
        reodom.pose.pose.position.z = msg.pose.pose.position.z + 0.5
        self.pub_cmd.publish(reodom)

if __name__ == '__main__':
   try:
      cmd = CMD()
   except rospy.ROSInterruptException:
      pass
