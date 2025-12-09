#!/usr/bin/env python3
"""
Mission scenario orchestrator that stitches line following, green-beacon navigation,
obstacle avoidance, and human-robot interaction into a single ROS 2 node.

The node progresses through three stages:
1. Follow a colored line until it disappears.
2. Search for a green beacon while avoiding obstacles with lidar.
3. Switch to the HRI gesture detector once the beacon fills enough of the image.
"""

import math
import os
import threading
import time
from enum import Enum
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan

import sdk.common as common
import sdk.pid as pid
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition

# ------------------
# Global constants
# ------------------
LINE_COLOR_KEY = "black"  # LAB name in the calibration YAML for the painted line
GREEN_COLOR_KEY = "green"  # LAB name in the calibration YAML for the target beacon
DEFAULT_BLACK_LAB = {"min": [0, 0, 0], "max": [40, 120, 120]}  # tuned for dark lines
LINE_SEARCH_LOST_FRAMES = 10  # frames without a contour before we give up on the line
BEACON_FOUND_AREA_RATIO = 0.06  # fraction of the image the green beacon should fill
MAX_SCAN_ANGLE = 240  # degree span to honor from the lidar
LINE_FORWARD_SPEED = 0.12  # m/s base forward speed while following the line
GREEN_FORWARD_SPEED = 0.10  # m/s base forward speed while chasing the beacon
ANGULAR_CLAMP = 1.2  # limit angular velocity to keep spins sane
LIDAR_STOP_DISTANCE = 0.25  # meters before we perform an immediate stop
LIDAR_AVOID_DISTANCE = 0.45  # meters when we start biasing away from obstacles
LIDAR_BIAS_GAIN = 0.8  # scale for converting lidar difference into angular velocity
VOICE_COOLDOWN = 8.0  # seconds between voice prompts in HRI stage


class MissionStage(Enum):
    """High-level phases of the mission."""

    LINE_FOLLOWING = 1
    GREEN_NAV = 2
    HRI = 3


class SimpleLineDetector:
    """Minimal line detector adapted from line_following.py focusing on bottom ROIs."""

    def __init__(self, rois: Tuple[Tuple[float, float, float, float, float], ...]):
        self.rois = rois
        self.weight_sum = sum(roi[-1] for roi in rois)

    @staticmethod
    def _max_contour(contours, threshold=30):
        contour_area = zip(contours, tuple(map(lambda c: abs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            return max(contour_area, key=lambda c_a: c_a[1])
        return None

    def __call__(self, image: np.ndarray, lowerb: Tuple[int, int, int], upperb: Tuple[int, int, int]):
        """Return deflection angle (radians) and debugging overlay."""
        h, w = image.shape[:2]
        centroid_sum = 0.0
        result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for roi in self.rois:
            blob = image[int(roi[0] * h): int(roi[1] * h), int(roi[2] * w): int(roi[3] * w)]
            img_lab = cv2.cvtColor(blob, cv2.COLOR_RGB2LAB)
            mask = cv2.inRange(img_lab, lowerb, upperb)
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            max_contour_area = self._max_contour(contours)
            if max_contour_area is not None:
                rect = cv2.minAreaRect(max_contour_area[0])
                box = np.intp(cv2.boxPoints(rect))
                for j in range(4):
                    box[j, 1] = box[j, 1] + int(roi[0] * h)
                cv2.drawContours(result_image, [box], -1, (0, 255, 255), 2)
                center_x = (box[0, 0] + box[2, 0]) / 2
                center_y = (box[0, 1] + box[2, 1]) / 2
                cv2.circle(result_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                centroid_sum += center_x * roi[-1]

        if centroid_sum == 0:
            return result_image, None
        center_pos = centroid_sum / max(self.weight_sum, 1e-6)
        deflection_angle = -math.atan((center_pos - (w / 2.0)) / (h / 2.0))
        return result_image, deflection_angle


class GreenBeaconDetector:
    """Simplified beacon detector from green_nav.py that tracks the largest green patch."""

    def __call__(self, image: np.ndarray, lowerb, upperb):
        result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (5, 5), 3)
        mask = cv2.inRange(img_blur, lowerb, upperb)
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
        contour_area = zip(contours, tuple(map(lambda c: abs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > 12, contour_area))
        if len(contour_area) == 0:
            return result_image, None, 0.0
        max_contour_area = max(contour_area, key=lambda c_a: c_a[1])
        rect = cv2.minAreaRect(max_contour_area[0])
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(result_image, [box], -1, (0, 255, 255), 2)
        center_x = (box[0, 0] + box[2, 0]) / 2
        center_y = (box[0, 1] + box[2, 1]) / 2
        cv2.circle(result_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        deflection_angle = -math.atan((center_x - (w / 2.0)) / (h / 2.0))
        area_ratio = max_contour_area[1] / max(float(w * h), 1e-6)
        return result_image, deflection_angle, area_ratio


class GestureClassifier:
    """Lightweight wrapper around MediaPipe gestures inspired by HRI.py."""

    def __init__(self):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils

    def classify(self, image: np.ndarray) -> str:
        results = self.hand_detector.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.multi_hand_landmarks:
            return ""
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        h, w, _ = image.shape

        def get_coord(idx):
            return np.array([hand_landmarks[idx].x * w, hand_landmarks[idx].y * h])

        wrist = get_coord(mp.solutions.hands.HandLandmark.WRIST)
        folded = 0
        extended = 0
        fingers_indices = [
            (mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.RING_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.PINKY_TIP, mp.solutions.hands.HandLandmark.PINKY_MCP),
            (mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.THUMB_CMC),
        ]
        for tip_idx, mcp_idx in fingers_indices:
            tip = get_coord(tip_idx)
            mcp = get_coord(mcp_idx)
            if np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist) * 1.1:
                folded += 1
            else:
                extended += 1
        if folded >= 4:
            return "fist"
        if extended >= 4:
            return "wave"
        return ""


class ScenarioNode(Node):
    """State machine combining line following, green beacon chase, and HRI gestures."""

    def __init__(self):
        super().__init__("scenario_runner")
        self.stage = MissionStage.LINE_FOLLOWING
        self.bridge = CvBridge()
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self._resolve_image_topic(), self.image_callback, qos)
        self.create_subscription(LaserScan, self._resolve_lidar_topic(), self.lidar_callback, qos)
        self.cmd_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.joints_pub = self.create_publisher(ServosPosition, "servo_controller", 1)

        # Helpers
        self.line_pid = pid.PID(0.030, 0.003, 0.0)
        self.beacon_pid = pid.PID(0.020, 0.003, 0.0)
        self.line_detector = SimpleLineDetector(((0.81, 0.83, 0, 1, 0.7), (0.69, 0.71, 0, 1, 0.2), (0.57, 0.59, 0, 1, 0.1)))
        self.green_detector = GreenBeaconDetector()
        self.gesture_classifier = GestureClassifier()

        # Calibration data
        lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")
        lab = lab_data.get("lab", {})
        camera_type = os.environ.get("DEPTH_CAMERA_TYPE", "ascamera")
        lookup_type = camera_type if camera_type in lab else "ascamera"
        self.line_color = lab.get(lookup_type, {}).get(LINE_COLOR_KEY, DEFAULT_BLACK_LAB)
        if self.line_color == DEFAULT_BLACK_LAB:
            self.get_logger().info("Using built-in black line profile; no picking required.")
        self.green_color = lab.get(lookup_type, {}).get(GREEN_COLOR_KEY, {"min": [0, 0, 0], "max": [255, 255, 255]})

        # State bookkeeping
        self.lidar_ranges: Optional[np.ndarray] = None
        self.lost_line_frames = 0
        self.last_beacon_area = 0.0
        self.last_voice_ts = 0.0
        self.heartbeats = 0

        threading.Thread(target=self._heartbeat, daemon=True).start()
        self.get_logger().info("Scenario orchestrator started. Beginning with line following.")

    # ------------------
    # Topic resolvers
    # ------------------
    def _resolve_image_topic(self) -> str:
        camera_type = os.environ.get("DEPTH_CAMERA_TYPE", "ascamera")
        topic_map = {
            "aurora": "/camera/image_raw",
            "ascamera": "/ascamera/camera_publisher/rgb0/image",
            "usb_cam": "/image_raw",
        }
        return topic_map.get(camera_type, "/ascamera/camera_publisher/rgb0/image")

    def _resolve_lidar_topic(self) -> str:
        lidar_type = os.environ.get("LIDAR_TYPE", "rplidar")
        return "/scan" if lidar_type else "/scan"

    # ------------------
    # Callbacks
    # ------------------
    def lidar_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        # ignore invalid ranges
        ranges = np.where(np.isfinite(ranges), ranges, np.inf)
        # Center crop to max scan angle
        total_angle = (len(ranges) - 1) * msg.angle_increment
        max_angle = math.radians(MAX_SCAN_ANGLE)
        if total_angle > max_angle:
            half_keep = int(max_angle / (2 * msg.angle_increment))
            center_idx = len(ranges) // 2
            ranges = ranges[center_idx - half_keep: center_idx + half_keep]
        self.lidar_ranges = ranges

    def image_callback(self, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        if self.stage == MissionStage.LINE_FOLLOWING:
            self._handle_line_following(image)
        elif self.stage == MissionStage.GREEN_NAV:
            self._handle_green_nav(image)
        else:
            self._handle_hri(image)

    # ------------------
    # Stage handlers
    # ------------------
    def _handle_line_following(self, image: np.ndarray):
        lowerb = tuple(self.line_color.get("min", [0, 0, 0]))
        upperb = tuple(self.line_color.get("max", [255, 255, 255]))
        _, deflection = self.line_detector(image, lowerb, upperb)
        if deflection is None:
            self.lost_line_frames += 1
            self.get_logger().debug("Line missing; incrementing lost frame count.")
            if self.lost_line_frames >= LINE_SEARCH_LOST_FRAMES:
                self._transition_to_green_nav(reason="Line lost")
            self._publish_twist(0.0, 0.0)
            return
        self.lost_line_frames = 0
        angular = float(self.line_pid.update(deflection))
        angular = max(min(angular, ANGULAR_CLAMP), -ANGULAR_CLAMP)
        self._publish_twist(LINE_FORWARD_SPEED, angular)

    def _handle_green_nav(self, image: np.ndarray):
        lowerb = tuple(self.green_color.get("min", [0, 0, 0]))
        upperb = tuple(self.green_color.get("max", [255, 255, 255]))
        _, deflection, area_ratio = self.green_detector(image, lowerb, upperb)
        self.last_beacon_area = area_ratio

        avoidance_bias = self._compute_avoidance_bias()
        if area_ratio >= BEACON_FOUND_AREA_RATIO:
            self.get_logger().info("Green beacon reached; entering HRI mode.")
            self._publish_twist(0.0, 0.0)
            self.stage = MissionStage.HRI
            return

        if deflection is None:
            # Lost beacon; slow spin to reacquire
            angular = 0.25 + avoidance_bias
            self._publish_twist(0.0, max(min(angular, ANGULAR_CLAMP), -ANGULAR_CLAMP))
            return

        angular = float(self.beacon_pid.update(deflection)) + avoidance_bias
        angular = max(min(angular, ANGULAR_CLAMP), -ANGULAR_CLAMP)
        self._publish_twist(GREEN_FORWARD_SPEED, angular)

    def _handle_hri(self, image: np.ndarray):
        gesture = self.gesture_classifier.classify(image)
        now = time.time()
        if gesture and (now - self.last_voice_ts) > VOICE_COOLDOWN:
            self.get_logger().info(f"Gesture detected: {gesture}")
            self.last_voice_ts = now
        if gesture == "wave":
            self._publish_twist(0.15, 0.0)
            self._set_head_pose("look_up")
        elif gesture == "fist":
            self._publish_twist(0.0, 0.0)
            self._set_head_pose("drive")

    # ------------------
    # Helpers
    # ------------------
    def _compute_avoidance_bias(self) -> float:
        if self.lidar_ranges is None:
            return 0.0
        left = np.mean(self.lidar_ranges[: len(self.lidar_ranges) // 3])
        right = np.mean(self.lidar_ranges[-len(self.lidar_ranges) // 3 :])
        front = np.min(self.lidar_ranges[len(self.lidar_ranges) // 3 : -len(self.lidar_ranges) // 3])
        if front < LIDAR_STOP_DISTANCE:
            self.get_logger().warn("Obstacle too close; stopping.")
            self._publish_twist(0.0, 0.0)
            return 0.0
        bias = 0.0
        if front < LIDAR_AVOID_DISTANCE:
            bias = (right - left) * LIDAR_BIAS_GAIN
            self.get_logger().debug(f"Avoidance bias applied: {bias:.2f}")
        return max(min(bias, ANGULAR_CLAMP), -ANGULAR_CLAMP)

    def _publish_twist(self, linear_x: float, angular_z: float):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def _set_head_pose(self, mode: str):
        if mode == "drive":
            positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 500))
        else:
            positions = ((10, 200), (5, 500), (4, 90), (3, 350), (2, 780), (1, 500))
        set_servo_position(self.joints_pub, 1.0, positions)

    def _transition_to_green_nav(self, reason: str):
        self.get_logger().info(f"Transitioning to green beacon search: {reason}")
        self.stage = MissionStage.GREEN_NAV
        self._publish_twist(0.0, 0.0)

    def _heartbeat(self):
        while rclpy.ok():
            self.heartbeats += 1
            self.get_logger().debug(f"Heartbeat {self.heartbeats}: stage={self.stage.name}")
            time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
