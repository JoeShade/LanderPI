#!/usr/bin/env python3
# encoding: utf-8
"""ROS 2 node that follows a green beacon with the RGB camera while dodging obstacles with lidar."""

import errno
import math
import os
import tempfile
import threading
import time
import cv2
import numpy as np
import rclpy
import sdk.common as common
import sdk.pid as pid
from app.common import Heart
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from interfaces.srv import SetFloat64
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ros_robot_controller_msgs.msg import PWMServoState, SetPWMServoState
from sensor_msgs.msg import Image, LaserScan
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition
from speech import speech
from std_srvs.srv import SetBool, Trigger


# ----------------------------
# Global configuration & flags
# ----------------------------
PROGRAM_NAME = "green_nav"
WINDOW_NAME = "green_nav"
WINDOW_LOCK_PATH = os.path.join(tempfile.gettempdir(), "green_nav_window.lock")
MAX_SCAN_ANGLE = 240  # degrees of the lidar scan considered for avoidance
DEFAULT_COLOR = "green"
DEFAULT_LOG_INTERVAL = 15  # frames between debug prints
DEFAULT_THRESHOLD = 0.6  # initial LAB tolerance for green detection
DEFAULT_STOP_THRESHOLD = 0.15  # meters before we issue an emergency stop
DEFAULT_TURN_SCALE = 0.5  # scales PID angular output
DEFAULT_BASE_FORWARD_SPEED = 0.15  # meters per second
DEFAULT_SEARCH_ANGULAR_SPEED = 0.2  # rad/s spin when looking for the beacon
DEFAULT_SEARCH_SPIN_IN_PLACE = True  # spin instead of creeping forward when searching
DEFAULT_VOICE_COOLDOWN = 15.0  # seconds between repeated voice prompts
VOICE_FEEDBACK_DEFAULT = True  # global flag so voice can be toggled from one place


# -------------
# Vision helper
# -------------
def _get_camera_type(default: str = "aurora") -> str:
    """Fetch DEPTH_CAMERA_TYPE with a sane default so missing envs don't crash."""

    camera_type = os.environ.get("DEPTH_CAMERA_TYPE") or os.environ.get("CAMERA_TYPE")
    if not camera_type:
        camera_type = default
        os.environ["DEPTH_CAMERA_TYPE"] = camera_type
    return camera_type


def _load_lab_config():
    """Resolve LAB config from env override or user home; return empty dict on failure."""
    candidates = []
    env_path = os.environ.get("LAB_CONFIG_PATH")
    if env_path:
        candidates.append(os.path.expanduser(env_path))
    candidates.append(os.path.expanduser("~/software/lab_tool/lab_config.yaml"))
    for path in candidates:
        if os.path.exists(path):
            try:
                return common.get_yaml_data(path)
            except Exception:
                break
    return {}


class LineFollower:
    """Detects the green target and returns how far off-center it sits."""

    def __init__(self, color, node):
        self.node = node
        self.min_contour_area = 12  # slightly more permissive to catch distant targets

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None

    def __call__(self, image, result_image, threshold, color):
        """Locate the largest green patch and report how far it sits from image center."""
        h, w = image.shape[:2]
        self.camera_type = _get_camera_type()
        if self.camera_type == 'ascamera':
            w = w + 200
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
            return result_image, None, 0.0

        rect = cv2.minAreaRect(max_contour_area[0])
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(result_image, [box], -1, (0, 255, 255), 2)

        # Use rectangle center as target centroid
        center_x = (box[0, 0] + box[2, 0]) / 2
        center_y = (box[0, 1] + box[2, 1]) / 2
        cv2.circle(result_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        deflection_angle = -math.atan((center_x - (w / 2.0)) / (h / 2.0))
        area_ratio = max_contour_area[1] / max(float(w * h), 1e-6)
        return result_image, deflection_angle, area_ratio


class GreenLineFollowingNode(Node):
    """ROS node orchestrating vision, lidar avoidance, and wheel commands."""

    # Prevent multiple OpenCV windows when multiple nodes are started.
    window_claimed = False
    window_lock_path = WINDOW_LOCK_PATH

    def __init__(self, name: str):
        if not rclpy.ok():
            rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # Identity and high-level state. Think of these flags as the simple
        # story of the robot: what color we are chasing, whether we are paused,
        # and how often we log what is happening.
        self.name = name
        self.color = DEFAULT_COLOR
        self.frame_count = 0
        self.log_interval = DEFAULT_LOG_INTERVAL

        # Run/stop flags adjusted by services
        self.is_running = False
        self.stop = False
        self.searching_for_green = True
        self.lost_frames = 0
        self.lost_frame_limit = 5
        self.count = 0  # lidar streak counter used for clearing stop states

        # Vision/PID helpers
        self.follower = None
        self.scan_angle = math.radians(45)
        self.pid = pid.PID(0.020, 0.003, 0.0)

        # Voice prompts and cooldowns
        self.voice_base = os.environ.get('VOICE_FEEDBACK_PATH') or os.path.join(os.path.dirname(__file__), 'feedback_voice')
        os.environ.setdefault('VOICE_FEEDBACK_PATH', self.voice_base)
        self.voice_enabled = bool(self.declare_parameter('voice_feedback', VOICE_FEEDBACK_DEFAULT).value)
        self.voice_cooldown = DEFAULT_VOICE_COOLDOWN
        self.last_voice_played = {}
        self.announced_search = False
        self.announced_acquired = False
        self.announced_avoidance = False

        # Search/target tuning; configurable via ROS parameters
        self.search_angular_speed = float(self.declare_parameter('search_angular_speed', DEFAULT_SEARCH_ANGULAR_SPEED).value)
        self.search_spin_in_place = bool(self.declare_parameter('search_spin_in_place', DEFAULT_SEARCH_SPIN_IN_PLACE).value)
        self.threshold = DEFAULT_THRESHOLD  # wider default tolerance for green
        self.stop_threshold = float(self.declare_parameter('stop_threshold', DEFAULT_STOP_THRESHOLD).value)
        self.turn_scale = float(self.declare_parameter('turn_scale', DEFAULT_TURN_SCALE).value)
        # Protect shared state touched by services and sensor callbacks.
        self.state_lock = threading.RLock()
        self.image_sub = None
        self.lidar_sub = None
        self.bridge = CvBridge()
        self.window_name = WINDOW_NAME
        self.window_initialized = False
        self.window_enabled = False
        self.window_lock_handle = None
        self.lab_data = _load_lab_config()
        self.camera_type = _get_camera_type()
        lab_map = self.lab_data.get('lab', {})
        self.lab_lookup_type = self.camera_type if self.camera_type in lab_map else 'ascamera'
        self.last_image_ts = None
        default_image_topic = self._resolve_image_topic()
        self.obstacle_avoidance_bias = 0.0
        self.avoidance_activation_distance = float(self.declare_parameter('avoidance_activation_distance', 0.50).value)
        self.avoidance_weight = float(self.declare_parameter('avoidance_weight', 0.8).value)
        self.max_avoidance_turn = float(self.declare_parameter('max_avoidance_turn', 0.8).value)
        self.avoidance_turn_in_place_gain = float(self.declare_parameter('avoidance_turn_in_place_gain', 2.5).value)
        self.min_avoidance_turn_in_place = float(self.declare_parameter('min_avoidance_turn_in_place', math.radians(80)).value)
        self.min_forward_after_probe = float(self.declare_parameter('min_forward_after_probe', 0.50).value)
        self.base_forward_speed = DEFAULT_BASE_FORWARD_SPEED
        self.emergency_retreat_distance = 0.20  # meters to back up on emergency stop
        self.emergency_stop_active = False
        self.emergency_retreat_until = None
        self.avoidance_engaged = False
        self.avoidance_in_progress = False  # lockout flag so avoidance doesn't retrigger while active
        self.avoidance_trigger_streak = int(self.declare_parameter('avoidance_trigger_streak', 2).value)
        self.obstacle_close_streak = 0
        self.current_area_ratio = 0.0
        self.near_target_area_ratio = float(self.declare_parameter('near_target_area_ratio', 0.08).value)
        self.near_target_activation_scale = float(self.declare_parameter('near_target_activation_scale', 0.5).value)
        self.near_target_bias_gain = float(self.declare_parameter('near_target_bias_gain', 3.0).value)
        self.target_reached_area_ratio = float(self.declare_parameter('target_reached_area_ratio', 0.06).value)
        self.target_spin_rate_multiplier = float(self.declare_parameter('target_spin_rate_multiplier', 1.0).value)
        self.target_spin_until = None
        self.target_spin_rate = 0.0
        self.last_avoidance_turn_sign = 1
        self.avoidance_side_hysteresis = float(self.declare_parameter('avoidance_side_hysteresis', 0.05).value)
        self.smoothed_avoidance_bias = 0.0
        self.avoidance_turn_in_place = False
        self.prev_avoidance_turn_in_place = False
        self.advance_after_probe_until = None
        self.min_front_distance = math.inf
        self.last_seen_green_ts = None
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
        # set_running is kept for compatibility but enter now starts navigation immediately.
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

    # -----------------------------
    # Logging and voice prompt tools
    # -----------------------------
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

    def _get_lab_config(self):
        """Return the LAB config for green, falling back across camera types."""
        lab_map = self.lab_data.get('lab', {})
        lab_config = lab_map.get(self.lab_lookup_type, {}).get(self.color)
        if lab_config is None and 'ascamera' in lab_map:
            lab_config = lab_map['ascamera'].get(self.color)
        if lab_config is None and lab_map:
            first_key = next(iter(lab_map))
            lab_config = lab_map[first_key].get(self.color)
        return lab_config

    def _handle_voice_prompts(self, searching_now, has_target, avoidance_now, just_entered_turn_in_place):
        """Centralize voice feedback so callers stay shorter and clearer."""
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

        if just_entered_turn_in_place and self.is_running:
            # Ensure a full stop before initiating turn-in-place avoidance.
            self.log_debug("[avoid] entering turn-in-place; sending stop before spin")
            self.mecanum_pub.publish(Twist())

    def _handle_target_reached(self, area_ratio, twist):
        """Stop the robot and schedule a celebratory spin when green fills the view."""
        if area_ratio < self.target_reached_area_ratio:
            return False
        self.is_running = False
        self.searching_for_green = False
        self.avoidance_engaged = False
        self.avoidance_in_progress = False
        self.avoidance_turn_in_place = False
        self.advance_after_probe_until = None
        self.obstacle_avoidance_bias = 0.0
        self.smoothed_avoidance_bias = 0.0
        self.mecanum_pub.publish(Twist())
        self.log_debug(f"Target reached: area_ratio={area_ratio:.3f} (threshold={self.target_reached_area_ratio})")
        # Schedule a quick 180-degree spin in place after reaching the target.
        turn_rate = self.max_avoidance_turn * self.avoidance_turn_in_place_gain * self.target_spin_rate_multiplier
        turn_rate = max(turn_rate, math.radians(90))
        spin_duration = math.pi / max(abs(turn_rate), 1e-3)
        self.target_spin_rate = turn_rate
        self._play_voice('success.wav')
        spin_start = time.time()
        self.target_spin_until = spin_start + spin_duration
        self.log_debug(f"Target spin scheduled: rate={turn_rate:.2f} rad/s, duration={spin_duration:.2f}s")
        return True

    # -----------------------------
    # Topic discovery and watchdogs
    # -----------------------------
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
        with self.state_lock:
            if self.is_running and self.searching_for_green and not self.stop:
                # Reminder while the robot spins to reacquire the beacon.
                self.log_debug(f"Searching for green target; angular z={self.search_angular_speed}")

    # -----------------------------
    # Servos and service endpoints
    # -----------------------------
    def pwm_controller(self, position_data):
        """Send a small list of servo positions to the controller."""
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
        """Start green navigation: reset PID, subscribe topics, and begin searching."""
        self.get_logger().info('\033[1;32m%s\033[0m' % "green_nav enter")
        if self.camera_type != 'ascamera':
            self.pwm_controller([1850, 1500])
        with self.state_lock:
            self.stop = False
            self.is_running = True  # Start navigation immediately on enter; no separate set_running needed.
            self.searching_for_green = True
            self.pid = pid.PID(1.1, 0.0, 0.0)
            self.follower = LineFollower([None, common.range_rgb[self.color]], self)
            self.threshold = 0.5
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
        """Stop navigation and tear down subscriptions so the robot holds still."""
        self.get_logger().info('\033[1;32m%s\033[0m' % "green_nav exit")
        with self.state_lock:
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
            self.is_running = False
            self.pid = pid.PID(0.00, 0.001, 0.0)
            self.follower = LineFollower([None, common.range_rgb[self.color]], self)
            self.threshold = 0.5
            self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        """Legacy toggle; prefer the enter/exit pair but keep this for compatibility."""
        # Deprecated: enter now starts navigation; this remains for compatibility.
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running (deprecated)")
        with self.state_lock:
            self.is_running = request.data
            if self.is_running:
                self.searching_for_green = True
            if not self.is_running:
                self.mecanum_pub.publish(Twist())
            self.log_debug(f"set_running called: is_running={self.is_running}, stop={self.stop}, searching_for_green={self.searching_for_green}")
        response.success = True
        response.message = "set_running"
        return response

    def set_threshold_srv_callback(self, request, response):
        """Adjust LAB detection tolerance at runtime; higher values accept more green."""
        self.get_logger().info('\033[1;32m%s\033[0m' % "set threshold")
        with self.state_lock:
            self.threshold = request.data
            self.log_debug(f"Threshold updated: {self.threshold}")
            response.success = True
            response.message = "set_threshold"
            return response

    # -----------------------------
    # Sensor callbacks
    # -----------------------------
    def lidar_callback(self, lidar_data):
        """Use the lidar scan to bias steering away from nearby obstacles."""
        with self.state_lock:
            # Focus on the forward arc and steer toward whichever side has more room.
            previous_turning_in_place = self.avoidance_turn_in_place
            avoidance_already_active = self.avoidance_in_progress
            just_triggered_avoidance = False
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
            activation_distance = self.avoidance_activation_distance
            if self.current_area_ratio >= self.near_target_area_ratio:
                activation_distance = max(self.avoidance_activation_distance * self.near_target_activation_scale, 0.05)
            obstacle_close = math.isfinite(min_front) and min_front < activation_distance
            if obstacle_close:
                self.obstacle_close_streak += 1
            else:
                self.obstacle_close_streak = 0
            if avoidance_already_active:
                # Do not retrigger; keep spinning until clear enough to drop lockout.
                if obstacle_close:
                    self.avoidance_engaged = True
                    self.avoidance_turn_in_place = True
                    self.log_debug(f"[avoid] locked; still too close (min_front={min_front:.2f} < {activation_distance})")
                else:
                    self.avoidance_in_progress = False
                    self.smoothed_avoidance_bias *= 0.5
                    self.log_debug("[avoid] clearance detected; unlocking avoidance")
            elif obstacle_close and self.obstacle_close_streak >= self.avoidance_trigger_streak:
                self.avoidance_engaged = True
                self.avoidance_turn_in_place = True
                self.avoidance_in_progress = True
                just_triggered_avoidance = True
                diff = (right_avg - left_avg)  # negative means obstacle is closer on the right
                if abs(diff) > self.avoidance_side_hysteresis:
                    self.last_avoidance_turn_sign = 1 if diff > 0 else -1
                # Keep turning the same way inside the hysteresis band to avoid oscillation.
                biased_diff = diff if abs(diff) > self.avoidance_side_hysteresis else self.last_avoidance_turn_sign * self.avoidance_side_hysteresis
                normalized = biased_diff / max(activation_distance, 1e-3)
                normalized = common.set_range(normalized, -1.0, 1.0)
                raw_bias = common.set_range(normalized, -self.max_avoidance_turn, self.max_avoidance_turn)
                if self.current_area_ratio >= self.near_target_area_ratio:
                    raw_bias *= self.near_target_bias_gain
                    raw_bias = common.set_range(raw_bias, -self.max_avoidance_turn, self.max_avoidance_turn)
                # Smooth bias to reduce fish-tailing.
                self.obstacle_avoidance_bias = 0.5 * self.smoothed_avoidance_bias + 0.5 * raw_bias
                self.smoothed_avoidance_bias = self.obstacle_avoidance_bias
                self.log_debug(f"[avoid] engaged: left={left_avg:.2f}, right={right_avg:.2f}, diff={diff:.2f}, bias={self.obstacle_avoidance_bias:.2f}, min_front={min_front:.2f}, activation={activation_distance}, hysteresis={self.avoidance_side_hysteresis}, turn_in_place={self.avoidance_turn_in_place}")
            elif obstacle_close:
                self.log_debug(f"[avoid] close but below streak: streak={self.obstacle_close_streak}/{self.avoidance_trigger_streak}, min_front={min_front:.2f}, activation={activation_distance}")
            elif math.isfinite(min_front):
                # Decay smoothed bias when not engaged.
                self.smoothed_avoidance_bias *= 0.5
                self.log_debug(f"[avoid] ahead but outside activation: left={left_avg:.2f}, right={right_avg:.2f}, min_front={min_front:.2f}, activation={activation_distance}")

            if just_triggered_avoidance:
                # Publish an immediate stop to cut motion as soon as avoidance is engaged.
                self.log_debug("[avoid] immediate stop issued on trigger")
                self.mecanum_pub.publish(Twist())

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
                    if not self.emergency_stop_active:
                        retreat_speed = max(self.base_forward_speed, 1e-3)
                        duration = self.emergency_retreat_distance / retreat_speed
                        self.emergency_retreat_until = time.time() + duration
                        self.emergency_stop_active = True
                        self.log_debug(f"[avoid] emergency stop: backing up {self.emergency_retreat_distance}m for {duration:.2f}s")
                    self.log_debug(f"Lidar stop triggered: left={min_dist_left:.2f}, right={min_dist_right:.2f}, threshold={self.stop_threshold}")
                else:
                    self.count += 1
                    if self.count > 5:
                        self.count = 0
                        if not self.emergency_stop_active:
                            self.stop = False
                            self.log_debug(f"Lidar clear: left={min_dist_left:.2f}, right={min_dist_right:.2f}")

    def image_callback(self, ros_image):
        """Blend camera target tracking with lidar avoidance and publish velocity commands."""
        # Convert ROS image message into a format OpenCV understands.
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        result_image = np.copy(rgb_image)
        self.last_image_ts = time.time()
        with self.state_lock:
            twist = Twist()
            now = time.time()
            if self.follower is None:
                self.follower = LineFollower([None, common.range_rgb[self.color]], self)
            base_speed = self.base_forward_speed  # forward speed base line
            # Start with forward motion, then blend in obstacle bias from lidar.
            twist.linear.x = base_speed
            avoid_correction = self.avoidance_weight * self.obstacle_avoidance_bias
            activation_distance = self.avoidance_activation_distance
            if self.current_area_ratio >= self.near_target_area_ratio:
                activation_distance = max(self.avoidance_activation_distance * self.near_target_activation_scale, 0.05)
            if self.avoidance_engaged:
                dist_scale = common.set_range(
                    (self.min_front_distance if math.isfinite(self.min_front_distance) else activation_distance)
                    / max(activation_distance, 1e-3),
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
            advance_active = bool(self.advance_after_probe_until and time.time() < self.advance_after_probe_until and not self.stop)
            if not advance_active:
                self.advance_after_probe_until = None
            retreat_active = False
            if self.emergency_stop_active:
                # Continue backing up until the retreat window ends.
                if self.emergency_retreat_until and now < self.emergency_retreat_until:
                    retreat_active = True
                else:
                    # Finish retreat: clear stop and avoidance lockouts.
                    self.emergency_stop_active = False
                    self.emergency_retreat_until = None
                    self.stop = False
                    self.avoidance_in_progress = False
                    self.avoidance_engaged = False
                    self.avoidance_turn_in_place = False
                    self.obstacle_avoidance_bias = 0.0
                    self.smoothed_avoidance_bias = 0.0
                    self.log_debug("[avoid] emergency retreat complete; resuming navigation")
            lab_config = self._get_lab_config()
            if lab_config is None:
                self.get_logger().error("LAB config missing for selected color; cannot proceed.")
                return

            # Run the vision detector and get steering hints.
            result_image, deflection_angle, area_ratio = self.follower(
                rgb_image,
                result_image,
                self.threshold,
                lab_config,
            )
            self.current_area_ratio = area_ratio or 0.0
            if self.avoidance_in_progress:
                # Ignore target tracking while avoidance is active to prevent conflicting navigation commands.
                deflection_angle = None
            if advance_active:
                # Hold navigation commands until post-avoidance advance finishes.
                deflection_angle = None
            if retreat_active:
                # Suppress navigation during emergency retreat.
                deflection_angle = None
            just_entered_turn_in_place = self.avoidance_turn_in_place and not self.prev_avoidance_turn_in_place
            self.prev_avoidance_turn_in_place = self.avoidance_turn_in_place
            if deflection_angle is not None:
                self.searching_for_green = False
                self.lost_frames = 0
                self.last_seen_green_ts = time.time()

            has_target = deflection_angle is not None
            searching_now = self.is_running and self.searching_for_green and not self.stop
            avoidance_now = self.is_running and self.avoidance_engaged

            self._handle_voice_prompts(searching_now, has_target, avoidance_now, just_entered_turn_in_place)
            spin_active = self.target_spin_until is not None and now < self.target_spin_until
            if not spin_active and self.target_spin_until is not None and now >= self.target_spin_until:
                # Finish spin: stop and clear.
                self.target_spin_until = None
                self.mecanum_pub.publish(Twist())
                self.log_debug("[target] spin complete; stopping.")
            if spin_active:
                twist.angular.z = self.target_spin_rate
                twist.linear.x = 0.0
                self.mecanum_pub.publish(twist)
            if retreat_active and self.is_running:
                twist.angular.z = 0.0
                twist.linear.x = -self.base_forward_speed
                self.mecanum_pub.publish(twist)
            elif advance_active and self.is_running and not self.stop:
                # Drive straight during post-avoidance advance; ignore navigation corrections.
                twist.angular.z = 0.0
                twist.linear.x = max(twist.linear.x, base_speed)
                self.mecanum_pub.publish(twist)
            elif deflection_angle is not None and self.is_running and not self.stop:
                if self._handle_target_reached(area_ratio, twist):
                    deflection_angle = None
                    has_target = False
                if deflection_angle is not None:
                    self.pid.update(deflection_angle)
                    pid_scale = 1.0
                    if self.avoidance_engaged:
                        pid_scale = 0.4 if self.avoidance_turn_in_place else 0.7
                        # Halt forward motion while obstacle avoidance is active.
                        twist.linear.x = 0.0
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
            if not self.window_initialized and not GreenLineFollowingNode.window_claimed:
                # Clear stale lockfiles so a crash doesn't block new windows.
                if os.path.exists(GreenLineFollowingNode.window_lock_path):
                    try:
                        mtime = os.path.getmtime(GreenLineFollowingNode.window_lock_path)
                        if (time.time() - mtime) > 300:
                            os.remove(GreenLineFollowingNode.window_lock_path)
                            self.log_debug("Removed stale window lockfile (>5m old).")
                    except Exception:
                        pass
                # Attempt to claim a cross-process lock to avoid duplicate windows.
                try:
                    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
                    self.window_lock_handle = os.open(GreenLineFollowingNode.window_lock_path, flags)
                    GreenLineFollowingNode.window_claimed = True
                    self.window_initialized = True
                    self.window_enabled = True
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    # Verify window creation immediately; if it fails, release the lock so another attempt can retry.
                    try:
                        cv2.imshow(self.window_name, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    except Exception as init_err:
                        self.get_logger().error(f"OpenCV window init failed; disabling display: {init_err}")
                        try:
                            cv2.destroyWindow(self.window_name)
                        except Exception:
                            pass
                        GreenLineFollowingNode.window_claimed = False
                        self.window_initialized = False
                        self.window_enabled = False
                        if self.window_lock_handle is not None:
                            try:
                                os.close(self.window_lock_handle)
                            except Exception:
                                pass
                            self.window_lock_handle = None
                        if os.path.exists(GreenLineFollowingNode.window_lock_path):
                            try:
                                os.remove(GreenLineFollowingNode.window_lock_path)
                            except Exception:
                                pass
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        self.log_debug("OpenCV window already claimed (lockfile exists); skipping duplicate window.")
                    else:
                        self.get_logger().error(f"Failed to claim window lock: {e}")
            if self.window_initialized and self.window_enabled:
                cv2.imshow(self.window_name, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"OpenCV display error: {e}")

        self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))


# -----------------------------
# Entrypoint
# -----------------------------
def main():
    """Spin the ROS node until the user exits."""
    node = GreenLineFollowingNode(PROGRAM_NAME)
    rclpy.spin(node)
    try:
        if node.window_initialized and node.window_enabled:
            cv2.destroyWindow(node.window_name)
            GreenLineFollowingNode.window_claimed = False
            try:
                if node.window_lock_handle is not None:
                    os.close(node.window_lock_handle)
                if os.path.exists(GreenLineFollowingNode.window_lock_path):
                    os.remove(GreenLineFollowingNode.window_lock_path)
            except Exception:
                pass
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
