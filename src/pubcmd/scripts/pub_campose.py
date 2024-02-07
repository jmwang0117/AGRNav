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
#import airsim
import tf
import math

def airpub():
    pub_pose = rospy.Publisher("/2dgazebo/camera_pose",PoseStamped,queue_size=1)
    rospy.init_node('pub_campose',anonymous=True)
    rate = rospy.Rate(30)

    listener = tf.TransformListener()
    count = 0
    while not rospy.is_shutdown():
         nowtime = rospy.Time.now()
         try:
            (trans,rot) = listener.lookupTransform('odom_combined', 'camera_depth_optical_frame', rospy.Time(0))
            #(trans,rot) = listener.lookupTransform('camera_link', 'camera_color_optical_frame', rospy.Time(0))
         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
             continue
        
         pose_msg = PoseStamped()
         pose_msg.header.seq = count
         count+=1
         pose_msg.header.frame_id = 'camera_depth_optical_frame'
         pose_msg.header.stamp = nowtime
         pose_msg.pose.position.x = trans[0]
         pose_msg.pose.position.y = trans[1]
         pose_msg.pose.position.z = trans[2]
     
         pos_x = pose_msg.pose.position.x
         pos_y = pose_msg.pose.position.y
         pos_z = pose_msg.pose.position.z


         pose_msg.pose.orientation.x = rot[0]
         pose_msg.pose.orientation.y = rot[1]
         pose_msg.pose.orientation.z = rot[2]
         pose_msg.pose.orientation.w = rot[3]

         #this is ENU

         

         pub_pose.publish(pose_msg)
         position = (pose_msg.pose.position.x,pose_msg.pose.position.y,pose_msg.pose.position.z)
         orientation = (pose_msg.pose.orientation.x,pose_msg.pose.orientation.y,pose_msg.pose.orientation.z,pose_msg.pose.orientation.w)



         print("//////////////////////////////")
         print("x ",pose_msg.pose.position.x)
         print("y ",pose_msg.pose.position.y)
         print("z ",pose_msg.pose.position.z)

         rate.sleep()

if __name__ == '__main__':
   try:
      airpub()
   except rospy.ROSInterruptException:
      pass
