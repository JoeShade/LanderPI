#!/usr/bin/env python3
# encoding: utf-8

import os
import math
import threading
import time
import cv2
import numpy as np
import rclpy
import sdk.pid as pid
import sdk.common as common
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import Image, LaserScan
from interfaces.srv import SetFloat64
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position
from speech import speech
from cv_bridge import CvBridge
from app.common import Heart


MAX_SCAN_ANGLE = 240  # degree (lidar scanning angle used for obstacle checks)


class LineFollower:
    """Minimal in-package copy of the line follower used by green_nav."""

    def __init__(self, color, node):
        self.node = node
        self.target_lab, self.target_rgb = color
        self.depth_camera_type = os.environ['DEPTH_CAMERA_TYPE']
        # Vertical stripe ROIs: left, center, right (full height), center weighted highest.
        self.rois = (
            (0.0, 1.0, 0.00, 0.33, 0.2),
            (0.0, 1.0, 0.33, 0.66, 0.6),
            (0.0, 1.0, 0.66, 1.00, 0.2),
        )

        self.weight_sum = sum(roi[-1] for roi in self.rois) or 1.0
        self.min_contour_area = 12  # slightly more permissive to catch distant targets

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None

    def __call__(self, image, result_image, threshold, color=None, use_color_picker=True):
        h, w = image.shape[:2]
        if os.environ['DEPTH_CAMERA_TYPE'] == 'ascamera':
            w = w + 200
        if use_color_picker:
            # Wider LAB tolerance for green detection.
            min_color = [int(self.target_lab[0] - 70 * threshold * 2),
                         int(self.target_lab[1] - 70 * threshold),
                         int(self.target_lab[2] - 70 * threshold)]
            max_color = [int(self.target_lab[0] + 70 * threshold * 2),
                         int(self.target_lab[1] + 70 * threshold),
                         int(self.target_lab[2] + 70 * threshold)]
            target_color = self.target_lab, min_color, max_color
            lowerb = tuple(target_color[1])
            upperb = tuple(target_color[2])
        else:
            lowerb = tuple(color['min'])
            upperb = tuple(color['max'])
        # Single global mask and single bounding box
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (5, 5), 3)
        mask = cv2.inRange(img_blur, lowerb, upperb)
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
        max_contour_area = self.get_area_max_contour(contours, self.min_contour_area)
        if max_contour_area is None:
            return result_image, None

        rect = cv2.minAreaRect(max_contour_area[0])
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(result_image, [box], -1, (0, 255, 255), 2)

        # Use rectangle center as target centroid
        center_x = (box[0, 0] + box[2, 0]) / 2
        center_y = (box[0, 1] + box[2, 1]) / 2
        cv2.circle(result_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        deflection_angle = -math.atan((center_x - (w / 2.0)) / (h / 2.0))
        return result_image, deflection_angle


class GreenLineFollowingNode(Node):
    #Line follower locked to a green target; color picker is disabled."

    def __init__(self, name: str):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.name = name
        self.color = "green"
        self.frame_count = 0
        self.log_interval = 15
        self.set_callback = False
        self.is_running = False
        self.follower = None
        self.scan_angle = math.radians(45)
        self.pid = pid.PID(0.020, 0.003, 0.0)
        self.empty = 0
        self.count = 0
        self.stop = False
        self.searching_for_green = True
        self.lost_frames = 0
        self.lost_frame_limit = 5
        self.language = os.environ.get('ASR_LANGUAGE', 'Chinese')
        # Keep green_nav audio alongside this file unless VOICE_FEEDBACK_PATH is set.
        self.voice_base = os.environ.get('VOICE_FEEDBACK_PATH') or os.path.join(os.path.dirname(__file__), 'feedback_voice')
        os.environ.setdefault('VOICE_FEEDBACK_PATH', self.voice_base)
        self.voice_enabled = bool(self.declare_parameter('voice_feedback', True).value)
        self.voice_cooldown = 15.0
        self.last_voice_played = {}
        self.announced_search = False
        self.announced_acquired = False
        self.announced_avoidance = False
        # Allow tuning how fast the robot spins while searching.
        self.search_angular_speed = float(self.declare_parameter('search_angular_speed', 0.2).value)
        # Whether to spin in place while searching (vs. slow turning).
        self.search_spin_in_place = bool(self.declare_parameter('search_spin_in_place', True).value)
        self.threshold = 0.6  # wider default tolerance for green
        # Stop only when obstacles are very close; configurable via parameter.
        self.stop_threshold = float(self.declare_parameter('stop_threshold', 0.15).value)
        # Scale turn rate toward the target for smoother steering.
        self.turn_scale = float(self.declare_parameter('turn_scale', 0.5).value)
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
        lab_map = self.lab_data.get('lab', {})
        self.lab_lookup_type = self.camera_type if self.camera_type in lab_map else 'ascamera'
        self.last_image_ts = None
        default_image_topic = self._resolve_image_topic()
        self.obstacle_avoidance_bias = 0.0
        self.avoidance_activation_distance = float(self.declare_parameter('avoidance_activation_distance', 0.50).value)
        self.avoidance_weight = float(self.declare_parameter('avoidance_weight', 0.8).value)
        self.max_avoidance_turn = float(self.declare_parameter('max_avoidance_turn', 0.8).value)
        self.avoidance_turn_in_place_gain = float(self.declare_parameter('avoidance_turn_in_place_gain', 2.5).value)
        self.min_avoidance_turn_in_place = float(self.declare_parameter('min_avoidance_turn_in_place', math.radians(60)).value)
        self.min_forward_after_probe = float(self.declare_parameter('min_forward_after_probe', 0.15).value)
        self.base_forward_speed = 0.15
        self.avoidance_engaged = False
        self.last_avoidance_turn_sign = 1
        self.avoidance_side_hysteresis = float(self.declare_parameter('avoidance_side_hysteresis', 0.05).value)
        self.smoothed_avoidance_bias = 0.0
        self.avoidance_turn_in_place = False
        self.prev_avoidance_turn_in_place = False
        self.advance_after_probe_until = None
        self.min_front_distance = math.inf
        # Handle auto-declared params (automatically_declare_parameters_from_overrides=True) without double-declare crashes.
        image_topic_param = self.get_parameter('image_topic')
        if image_topic_param.type_ == Parameter.Type.NOT_SET or image_topic_param.value is None:
            try:
                self.declare_parameter('image_topic', default_image_topic)
            except Exception:
                pass
            self.image_topic = default_image_topic
        else:
            self.image_topic = image_topic_param.value
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
        self.create_timer(5.0, self._image_watchdog)
        self.create_timer(2.0, self._search_status_tick)

        Heart(self, self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))
        self.debug = bool(self.get_parameter('debug').value)
        self.log_debug(f"Debug logging enabled. DEPTH_CAMERA_TYPE={self.camera_type}, LIDAR_TYPE={self.lidar_type}, MACHINE_TYPE={self.machine_type}")
        self.log_debug(f"Stop threshold set to {self.stop_threshold} meters (adjust with parameter stop_threshold)")
        self.log_debug(f"Turn scale set to {self.turn_scale} (adjust with parameter turn_scale)")
        self.get_logger().info('\033[1;32m%s\033[0m' % 'green_nav start')

    def log_debug(self, message: str):
        if self.debug:
            # rclpy logger already prints to terminal; keep messages concise.
            self.get_logger().info(f"[debug] {message}")

    def _voice_path(self, name: str) -> str:
        """Resolve voice file path in the single feedback_voice directory."""
        base = self.voice_base
        filename = name if os.path.splitext(os.path.basename(name))[1] else name + '.wav'
        if os.path.isabs(filename):
            return filename
        return os.path.join(base, filename)

    def _play_voice(self, name: str, volume: int = 80):
        """Lightweight inlined audio playback so we do not depend on voice_play."""
        if not self.voice_enabled:
            return
        path = self._voice_path(name)
        now = time.time()
        last_played = self.last_voice_played.get(path)
        if last_played is not None and (now - last_played) < self.voice_cooldown:
            remaining = self.voice_cooldown - (now - last_played)
            self.log_debug(f"Voice playback skipped for {path}; {remaining:.1f}s cooldown remaining.")
            return
        try:
            speech.set_volume(volume)
            speech.play_audio(path)
            self.last_voice_played[path] = now
        except Exception as e:
            self.get_logger().error(f"Voice playback failed for {name}: {e}")

    def _resolve_image_topic(self) -> str:
        if self.camera_type == 'aurora':
            # On this platform Aurora images are published under ascamera namespace
            return '/ascamera/camera_publisher/rgb0/image'
        if self.camera_type == 'usb_cam':
            return '/camera/image'
        return '/ascamera/camera_publisher/rgb0/image'

    def _image_watchdog(self):
        if not self.debug:
            return
        now = time.time()
        if self.last_image_ts is None:
            self.log_debug("Waiting for first image on topic: " + self.image_topic + " (override with parameter image_topic)")
        elif now - self.last_image_ts > 5.0:
            self.log_debug(f"No images received for {now - self.last_image_ts:.1f}s on {self.image_topic} (override with parameter image_topic)")

    def _search_status_tick(self):
        if not self.debug:
            return
        with self.lock:
            if self.is_running and self.searching_for_green and not self.stop:
                self.log_debug(f"Searching for green… angular z={self.search_angular_speed}")

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
            self.log_debug("Entering green_nav: reset PID and thresholds; creating subscriptions if needed.")
            if self.image_sub is None:
                image_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
                self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile=image_qos)
                self.log_debug(f"Subscribed to image topic: {self.image_topic}")
            if self.lidar_sub is None:
                qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
                self.lidar_sub = self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, qos)
                set_servo_position(self.joints_pub, 1, ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 500))) # Pitched robot arm up to see green beacon
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
            self.log_debug("Exit: subscriptions destroyed and robot stopped.")
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
            self.log_debug(f"set_running called: is_running={self.is_running}, stop={self.stop}, searching_for_green={self.searching_for_green}")
        response.success = True
        response.message = "set_running"
        return response

    def set_threshold_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set threshold")
        with self.lock:
            self.threshold = request.data
            self.log_debug(f"Threshold updated: {self.threshold}")
            response.success = True
            response.message = "set_threshold"
            return response

    def lidar_callback(self, lidar_data):
        with self.lock:
            previous_turning_in_place = self.avoidance_turn_in_place
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

            # Obstacle avoidance bias: steer toward the side with more free space when something is between us and the target.
            self.obstacle_avoidance_bias = 0.0
            self.avoidance_engaged = False
            self.avoidance_turn_in_place = False
            self.min_front_distance = math.inf
            if len(min_dist_left_) > 0 and len(min_dist_right_) > 0:
                left_window = left_range[left_nonzero][left_nonan]
                right_window = right_range[right_nonzero][right_nonan]
                left_avg = float(np.median(left_window)) if len(left_window) > 0 else math.inf
                right_avg = float(np.median(right_window)) if len(right_window) > 0 else math.inf
                min_front = min(left_avg, right_avg)
                self.min_front_distance = min_front
                if math.isfinite(min_front) and min_front < self.avoidance_activation_distance:
                    self.avoidance_engaged = True
                    diff = (right_avg - left_avg)  # negative means obstacle is closer on the right
                    if abs(diff) > self.avoidance_side_hysteresis:
                        self.last_avoidance_turn_sign = 1 if diff > 0 else -1
                    # Keep turning the same way inside the hysteresis band to avoid oscillation.
                    biased_diff = diff if abs(diff) > self.avoidance_side_hysteresis else self.last_avoidance_turn_sign * self.avoidance_side_hysteresis
                    normalized = biased_diff / max(self.avoidance_activation_distance, 1e-3)
                    normalized = common.set_range(normalized, -1.0, 1.0)
                    raw_bias = common.set_range(normalized, -self.max_avoidance_turn, self.max_avoidance_turn)
                    # Smooth bias to reduce fish-tailing.
                    self.obstacle_avoidance_bias = 0.5 * self.smoothed_avoidance_bias + 0.5 * raw_bias
                    self.smoothed_avoidance_bias = self.obstacle_avoidance_bias
                    if min_front < self.avoidance_activation_distance * 0.8:
                        self.avoidance_turn_in_place = True
                    self.log_debug(f"Obstacle avoidance engaged: left={left_avg:.2f}, right={right_avg:.2f}, diff={diff:.2f}, bias={self.obstacle_avoidance_bias:.2f}, min_front={min_front:.2f}, activation={self.avoidance_activation_distance}, hysteresis={self.avoidance_side_hysteresis}, turn_in_place={self.avoidance_turn_in_place}")
                elif math.isfinite(min_front):
                    # Decay smoothed bias when not engaged.
                    self.smoothed_avoidance_bias *= 0.5
                    self.log_debug(f"Obstacle ahead but outside activation: left={left_avg:.2f}, right={right_avg:.2f}, activation={self.avoidance_activation_distance}")

            if self.avoidance_turn_in_place:
                # Clear any pending forward-advance window while still probing.
                self.advance_after_probe_until = None
            elif previous_turning_in_place and not self.avoidance_turn_in_place:
                duration = max(self.min_forward_after_probe / max(self.base_forward_speed, 1e-3), 0.2)
                self.advance_after_probe_until = time.time() + duration
                self.log_debug(f"Finished turn-in-place; advancing for {duration:.2f}s to clear obstacle.")

            # Stop handling after avoidance assessment so avoidance can engage first.
            if len(min_dist_left_) > 1 and len(min_dist_right_) > 1:
                min_dist_left = min_dist_left_.min()
                min_dist_right = min_dist_right_.min()
                if min_dist_left < self.stop_threshold or min_dist_right < self.stop_threshold:
                    self.stop = True
                    self.log_debug(f"Lidar stop triggered: left={min_dist_left:.2f}, right={min_dist_right:.2f}, threshold={self.stop_threshold}")
                else:
                    self.count += 1
                    if self.count > 5:
                        self.count = 0
                        self.stop = False
                        self.log_debug(f"Lidar clear: left={min_dist_left:.2f}, right={min_dist_right:.2f}")

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        self.image_height, self.image_width = rgb_image.shape[:2]
        result_image = np.copy(rgb_image)
        self.last_image_ts = time.time()
        with self.lock:
            twist = Twist()
            if self.follower is None:
                self.follower = LineFollower([None, common.range_rgb[self.color]], self)
            base_speed = self.base_forward_speed  # Speed variable
            twist.linear.x = base_speed
            avoid_correction = self.avoidance_weight * self.obstacle_avoidance_bias
            if self.avoidance_engaged:
                dist_scale = common.set_range(
                    (self.min_front_distance if math.isfinite(self.min_front_distance) else self.avoidance_activation_distance)
                    / max(self.avoidance_activation_distance, 1e-3),
                    0.1,
                    1.0,
                )
                twist.linear.x = base_speed * dist_scale
                if self.avoidance_turn_in_place:
                    twist.linear.x = 0.0
            elif abs(avoid_correction) > 1e-3:
                twist.linear.x *= 0.6  # slow down while maneuvering around an obstacle
            if self.advance_after_probe_until and time.time() < self.advance_after_probe_until and not self.stop:
                twist.linear.x = max(twist.linear.x, base_speed)
            lab_map = self.lab_data.get('lab', {})
            # Robust LAB selection with fallback to first available entry
            lab_config = lab_map.get(self.lab_lookup_type, {}).get(self.color)
            if lab_config is None and 'ascamera' in lab_map:
                lab_config = lab_map['ascamera'].get(self.color)
            if lab_config is None and lab_map:
                first_key = next(iter(lab_map))
                lab_config = lab_map[first_key].get(self.color)
            if lab_config is None:
                self.get_logger().error("LAB config missing for selected color; cannot proceed.")
                return

            result_image, deflection_angle = self.follower(
                rgb_image,
                result_image,
                self.threshold,
                lab_config,
                False,
            )
            if deflection_angle is not None:
                self.searching_for_green = False
                self.lost_frames = 0
                self.last_seen_green_ts = time.time()

            has_target = deflection_angle is not None
            searching_now = self.is_running and self.searching_for_green and not self.stop
            avoidance_now = self.is_running and self.avoidance_engaged

            if searching_now and not self.announced_search:
                self._play_voice('start_track_green')
                self.announced_search = True
                self.announced_acquired = False
            elif not searching_now:
                self.announced_search = False

            if has_target and self.is_running and not self.announced_acquired:
                self._play_voice('find_target')
                self.announced_acquired = True
                self.announced_search = False

            if avoidance_now and not self.announced_avoidance:
                self._play_voice('warning')
                self.announced_avoidance = True
            elif not avoidance_now:
                self.announced_avoidance = False
            if deflection_angle is not None and self.is_running and not self.stop:
                self.pid.update(deflection_angle)
                pid_scale = 1.0
                if self.avoidance_engaged:
                    pid_scale = 0.4 if self.avoidance_turn_in_place else 0.7
                if 'Acker' in self.machine_type:
                    steering_angle = common.set_range(-self.pid.output, -math.radians(40), math.radians(40))
                    if steering_angle != 0:
                        R = 0.145 / math.tan(steering_angle)
                        twist.angular.z = self.turn_scale * (twist.linear.x / R) * pid_scale
                else:
                    twist.angular.z = self.turn_scale * common.set_range(-self.pid.output, -1.0, 1.0) * pid_scale
                twist.angular.z += common.set_range(avoid_correction, -self.max_avoidance_turn, self.max_avoidance_turn)
                if self.avoidance_turn_in_place:
                    twist.linear.x = 0.0
                    turn_rate = self.max_avoidance_turn * self.avoidance_turn_in_place_gain
                    turn_rate = max(turn_rate, self.min_avoidance_turn_in_place)
                    twist.angular.z = turn_rate * self.last_avoidance_turn_sign
                    self.log_debug(f"Turning in place to avoid obstacle; angular={twist.angular.z:.2f}, front={self.min_front_distance:.2f}")
                self.mecanum_pub.publish(twist)
            elif self.is_running and self.searching_for_green and not self.stop:
                # Force spin-in-place while searching so the robot doesn't creep forward.
                if self.search_spin_in_place:
                    twist.linear.x = 0.0
                    twist.linear.y = 0.0
                twist.angular.z = self.search_angular_speed + common.set_range(avoid_correction, -self.max_avoidance_turn, self.max_avoidance_turn)
                if self.avoidance_turn_in_place:
                    turn_rate = self.max_avoidance_turn * self.avoidance_turn_in_place_gain
                    turn_rate = max(turn_rate, self.min_avoidance_turn_in_place)
                    twist.angular.z = turn_rate * self.last_avoidance_turn_sign
                    self.log_debug(f"Searching turn-in-place to probe obstacle clearance; angular={twist.angular.z:.2f}, front={self.min_front_distance:.2f}")
                self.mecanum_pub.publish(twist)
            elif self.is_running and not self.stop:
                # Lost the target: stop previous twist so we don't keep spinning blindly.
                self.lost_frames += 1
                if not self.searching_for_green and self.lost_frames >= self.lost_frame_limit:
                    self.mecanum_pub.publish(Twist())  # stop motion
                    self.searching_for_green = True
                    self.lost_frames = 0
                    self.log_debug("Lost green target: stopping motion and re-entering search mode")
            elif self.stop:
                self.mecanum_pub.publish(Twist())
            else:
                self.pid.clear()

            self.frame_count += 1
            if self.frame_count % self.log_interval == 0:
                pid_output = getattr(self.pid, 'output', None)
                pid_output_str = f"{pid_output:.3f}" if isinstance(pid_output, (int, float)) else "n/a"
                self.log_debug(f"Frame {self.frame_count}: running={self.is_running}, stop={self.stop}, searching={self.searching_for_green}, deflection={deflection_angle}, pid_out={pid_output_str}")
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
