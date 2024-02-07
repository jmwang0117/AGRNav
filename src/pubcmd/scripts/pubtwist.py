#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from quadrotor_msgs.msg import PositionCommand
import math

class CMD:
   def __init__(self):
      rospy.init_node('pub_twist', anonymous=True)
      self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
      self.sub_cmd = rospy.Subscriber("/planning/pos_cmd", PositionCommand, self.cmd_cb)
      self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odom_cb)

      self.target_x = 0.0
      self.target_y = 0.0
      self.target_yaw = 0.0

      self.current_x = 0.0
      self.current_y = 0.0
      self.current_yaw = 0.0
      self.rate = rospy.Rate(30)  # 控制循环的频率为10Hz
        
      self.integral_error = 0.0
      self.previous_error = 0.0
      self.previous_time = rospy.get_time()

      self.velocity = None
      self.acceleration = None

      while not rospy.is_shutdown():
          self.run()
          self.rate.sleep()
          
   def pid_controller(self, error, dt):
      Kp = 0.75  # 比例增益
      Ki = 0.1   # 积分增益
      Kd = 0.05  # 微分增益

      self.integral_error += error * dt
      derivative_error = (error - self.previous_error) / dt
      self.previous_error = error

      output = Kp * error + Ki * self.integral_error + Kd * derivative_error
      return output

   def cmd_cb(self, msg):
      self.target_x = msg.position.x
      self.target_y = msg.position.y
      self.target_yaw = msg.yaw
      self.velocity = msg.velocity
      self.acceleration = msg.acceleration

   def odom_cb(self, msg):
      self.current_x = msg.pose.pose.position.x
      self.current_y = msg.pose.pose.position.y
      q_x = msg.pose.pose.orientation.x
      q_y = msg.pose.pose.orientation.y
      q_z = msg.pose.pose.orientation.z
      q_w = msg.pose.pose.orientation.w
      _, _, self.current_yaw = self.quaternion_to_euler(q_x, q_y, q_z, q_w)

   def quaternion_to_euler(self, x, y, z, w):
      t0 = +2.0 * (w * x + y * z)
      t1 = +1.0 - 2.0 * (x * x + y * y)
      roll = math.atan2(t0, t1)

      t2 = +2.0 * (w * y - z * x)
      t2 = +1.0 if t2 > +1.0 else t2
      t2 = -1.0 if t2 < -1.0 else t2
      pitch = math.asin(t2)

      t3 = +2.0 * (w * z + x * y)
      t4 = +1.0 - 2.0 * (y * y + z * z)
      yaw = math.atan2(t3, t4)

      return roll, pitch, yaw

   def run(self):
      if self.velocity is not None and self.acceleration is not None:
         if not self.is_velocity_zero(self.velocity) or not self.is_acceleration_zero(self.acceleration):
               dx = self.target_x - self.current_x
               dy = self.target_y - self.current_y               
               dL = math.sqrt(dx * dx + dy * dy)
               cmd_command = Twist()
               cmd_command.linear.x = 0.5 * dx
               cmd_command.linear.y = 0.5 * dy

               if dL > 0.1:
                  dyaw = math.atan2(dy, dx) - self.current_yaw

                  if dyaw > 1.5 * math.pi:
                     dyaw = dyaw - 2 * math.pi
                  elif dyaw < -1.5 * math.pi:
                     dyaw = dyaw + 2 * math.pi

                  current_time = rospy.get_time()
                  dt = current_time - self.previous_time
                  self.previous_time = current_time
                  
                  cmd_command.angular.z = self.pid_controller(dyaw, dt)
               else:
                  cmd_command.angular.z = 0

               if dL < 0.02:
                  cmd_command.linear.x = 0.0
                  cmd_command.linear.y = 0.0
                  cmd_command.angular.z = 0.0

               self.pub_cmd.publish(cmd_command)  # 发布控制指令到/cmd_vel话题
               rospy.loginfo("cmd_linear_x: %f", cmd_command.linear.x)  # 打印线速度值
               rospy.loginfo("cmd_linear_y: %f", cmd_command.linear.y)  # 打印线速度值
               rospy.loginfo("cmd_angular_z: %f", cmd_command.angular.z)  # 打印角速度值
         else:
               cmd_command = Twist()
               self.pub_cmd.publish(cmd_command)  # 发布零速度和零角速度到/cmd_vel话题

        
   def is_velocity_zero(self, velocity):
      return (
         abs(velocity.x) < 0.01 and
         abs(velocity.y) < 0.01 and
         abs(velocity.z) < 0.01
      )

   def is_acceleration_zero(self, acceleration):
      return (
         abs(acceleration.x) < 0.01 and
         abs(acceleration.y) < 0.01 and
         abs(acceleration.z) < 0.01
      )

if __name__ == '__main__':
    try:
        cmd = CMD()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass