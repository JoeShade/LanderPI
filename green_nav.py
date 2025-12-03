#!/usr/bin/env python3
# encoding: utf-8

import os
import math
import threading
import cv2
import numpy as np
import rclpy
import sdk.pid as pid
import sdk.common as common
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import Image, LaserScan
from interfaces.srv import SetFloat64
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position
from cv_bridge import CvBridge
from app.common import Heart
from line_following import LineFollower, MAX_SCAN_ANGLE


class GreenLineFollowingNode(Node):
    #Line follower locked to a green target; color picker is disabled."

    def __init__(self, name: str):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.name = name
        self.color = "green"
        self.set_callback = False
        self.is_running = False
        self.follower = None
        self.scan_angle = math.radians(45)
        self.pid = pid.PID(0.030, 0.003, 0.0)
        self.empty = 0
        self.count = 0
        self.stop = False
        self.searching_for_green = True
        self.search_angular_speed = 0.4
        self.threshold = 0.5
        self.stop_threshold = 0.4
        self.lock = threading.RLock()
        self.image_sub = None
        self.lidar_sub = None
        self.image_height = None
        self.image_width = None
        self.bridge = CvBridge()
        self.window_name = "green_nav"
        self.window_initialized = False
        self.use_color_picker = False  # lock to green
        self.lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")
        self.camera_type = os.environ['DEPTH_CAMERA_TYPE']
        self.lidar_type = os.environ.get('LIDAR_TYPE')
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.pwm_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 10)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        self.create_service(SetFloat64, '~/set_threshold', self.set_threshold_srv_callback)
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)

        Heart(self, self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))
        self.debug = self.get_parameter('debug').value
        self.get_logger().info('\033[1;32m%s\033[0m' % 'green_nav start')

    def pwm_controller(self, position_data):
        pwm_list = []
        msg = SetPWMServoState()
        msg.duration = 0.2
        for i in range(len(position_data)):
            pos = PWMServoState()
            pos.id = [i + 1]
            pos.position = [int(position_data[i])]
            pwm_list.append(pos)
        msg.state = pwm_list
        self.pwm_pub.publish(msg)

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "green_nav enter")
        if os.environ['DEPTH_CAMERA_TYPE'] != 'ascamera':
            self.pwm_controller([1850, 1500])
        with self.lock:
            self.stop = False
            self.is_running = False
            self.searching_for_green = True
            self.pid = pid.PID(1.1, 0.0, 0.0)
            self.follower = LineFollower([None, common.range_rgb[self.color]], self)
            self.threshold = 0.5
            self.empty = 0
            if self.image_sub is None:
                self.image_sub = self.create_subscription(Image, 'ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
            if self.lidar_sub is None:
                qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
                self.lidar_sub = self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, qos)
                set_servo_position(self.joints_pub, 1, ((10, 200), (5, 500), (4, 90), (3, 150), (2, 770), (1, 500))) # Pitched robot arm up to see green beacon
            self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "green_nav exit")
        try:
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
            if self.lidar_sub is not None:
                self.destroy_subscription(self.lidar_sub)
                self.lidar_sub = None
        except Exception as e:
            self.get_logger().error(str(e))
        with self.lock:
            self.is_running = False
            self.pid = pid.PID(0.00, 0.001, 0.0)
            self.follower = LineFollower([None, common.range_rgb[self.color]], self)
            self.threshold = 0.5
            self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.is_running = request.data
            self.empty = 0
            if self.is_running:
                self.searching_for_green = True
            if not self.is_running:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def set_threshold_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set threshold")
        with self.lock:
            self.threshold = request.data
            response.success = True
            response.message = "set_threshold"
            return response

    def lidar_callback(self, lidar_data):
        if self.lidar_type != 'G4':
            min_index = int(math.radians(MAX_SCAN_ANGLE / 2.0) / lidar_data.angle_increment)
            max_index = int(math.radians(MAX_SCAN_ANGLE / 2.0) / lidar_data.angle_increment)
            left_ranges = lidar_data.ranges[:max_index]
            right_ranges = lidar_data.ranges[::-1][:max_index]
        elif self.lidar_type == 'G4':
            min_index = int(math.radians((360 - MAX_SCAN_ANGLE) / 2.0) / lidar_data.angle_increment)
            max_index = int(math.radians(180) / lidar_data.angle_increment)
            left_ranges = lidar_data.ranges[min_index:max_index][::-1]
            right_ranges = lidar_data.ranges[::-1][min_index:max_index][::-1]

        angle = self.scan_angle / 2
        angle_index = int(angle / lidar_data.angle_increment + 0.50)
        left_range, right_range = np.array(left_ranges[:angle_index]), np.array(right_ranges[:angle_index])

        left_nonzero = left_range.nonzero()
        right_nonzero = right_range.nonzero()
        left_nonan = np.isfinite(left_range[left_nonzero])
        right_nonan = np.isfinite(right_range[right_nonzero])
        min_dist_left_ = left_range[left_nonzero][left_nonan]
        min_dist_right_ = right_range[right_nonzero][right_nonan]
        if len(min_dist_left_) > 1 and len(min_dist_right_) > 1:
            min_dist_left = min_dist_left_.min()
            min_dist_right = min_dist_right_.min()
            if min_dist_left < self.stop_threshold or min_dist_right < self.stop_threshold:
                self.stop = True
            else:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.stop = False

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        self.image_height, self.image_width = rgb_image.shape[:2]
        result_image = np.copy(rgb_image)
        with self.lock:
            twist = Twist()
            if self.follower is None:
                self.follower = LineFollower([None, common.range_rgb[self.color]], self)
            twist.linear.x = 0.15 # Speed variable
            result_image, deflection_angle = self.follower(
                rgb_image,
                result_image,
                self.threshold,
                self.lab_data['lab'][self.camera_type][self.color],
                False,
            )
            if deflection_angle is not None:
                self.searching_for_green = False
            if deflection_angle is not None and self.is_running and not self.stop:
                self.pid.update(deflection_angle)
                if 'Acker' in self.machine_type:
                    steering_angle = common.set_range(-self.pid.output, -math.radians(40), math.radians(40))
                    if steering_angle != 0:
                        R = 0.145 / math.tan(steering_angle)
                        twist.angular.z = twist.linear.x / R
                else:
                    twist.angular.z = common.set_range(-self.pid.output, -1.0, 1.0)
                self.mecanum_pub.publish(twist)
            elif self.is_running and self.searching_for_green and not self.stop:
                twist.linear.x = 0.0
                twist.angular.z = self.search_angular_speed
                self.mecanum_pub.publish(twist)
            elif self.stop:
                self.mecanum_pub.publish(Twist())
            else:
                self.pid.clear()
        # Show live camera view in an OpenCV window
        try:
            if not self.window_initialized:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.window_initialized = True
            cv2.imshow(self.window_name, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"OpenCV display error: {e}")

        self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))


def main():
    node = GreenLineFollowingNode('green_nav')
    rclpy.spin(node)
    try:
        if node.window_initialized:
            cv2.destroyWindow(node.window_name)
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
