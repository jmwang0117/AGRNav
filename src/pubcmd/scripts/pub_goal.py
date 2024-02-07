#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import rospy
import roslib
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np

import tf
import math
#import airsim


class CMD:

    def __init__(self):
        rospy.init_node('pub_goal',anonymous=True)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal",PoseStamped,queue_size=1)
        self.goal_sub = rospy.Subscriber("/odom_car", Odometry, self.goal_cb)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_cb)
        self.odom = Odometry()


        rospy.spin()
    
    def odom_cb(self,msg):
        self.odom = msg
    
    def goal_cb(self, msg):
        goal_point = PoseStamped()
        goal_point.header = msg.header
        
        x1 = msg.pose.pose.position.x
        y1 = msg.pose.pose.position.y
        x2 = self.odom.pose.pose.position.x
        y2 = self.odom.pose.pose.position.y

        dx = x1 - x2
        dy = y1 - y2
        dis = math.sqrt(dx*dx + dy*dy)
        l = 2

        goal_point.pose.position.x = (1 - l/dis)*x1 + l/dis*x2
        goal_point.pose.position.y = (1 - l/dis)*y1 + l/dis*y2

        
        if(dis > 1+l):
           self.goal_pub.publish(goal_point)


if __name__ == '__main__':
   try:
      cmd = CMD()
   except rospy.ROSInterruptException:
      pass
