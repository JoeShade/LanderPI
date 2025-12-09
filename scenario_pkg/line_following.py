#!/usr/bin/env python3
# encoding: utf-8
"""Follow a colored line while avoiding obstacles with lidar.

The node keeps the explanation light and conversational so newcomers can see
how camera processing, lidar checks, and wheel commands fit together.
"""

# line following
import os
import cv2
import math
import time
import rclpy
import queue
import threading
import numpy as np
import sdk.pid as pid
import sdk.common as common
from rclpy.node import Node
from app.common import Heart
from cv_bridge import CvBridge
from app.common import ColorPicker
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import Image, LaserScan
from interfaces.srv import SetPoint, SetFloat64, SetString
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ros_robot_controller_msgs.msg import MotorsState, SetPWMServoState, PWMServoState
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position
from scenario_pkg.roi_config import get_rois


MAX_SCAN_ANGLE = 240  # degree(the scanning angle of lidar. The covered part is always eliminated)


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
    """Lightweight helper that finds the colored strip in the camera image."""

    def __init__(self, color, node):
        self.node = node
        self.target_lab, self.target_rgb = color
        self.depth_camera_type = _get_camera_type()
        self.rois = get_rois(self.depth_camera_type)
        self.weight_sum = sum(roi[-1] for roi in self.rois)

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        """Return the largest contour above the given area threshold."""
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None

    def __call__(self, image, result_image, threshold, color=None, use_color_picker=True):
        centroid_sum = 0
        h, w = image.shape[:2]
        if self.depth_camera_type == 'ascamera':
            w = w + 200
        hit_count = 0
        if use_color_picker:
            min_color = [int(self.target_lab[0] - 50 * threshold * 2),
                         int(self.target_lab[1] - 50 * threshold),
                         int(self.target_lab[2] - 50 * threshold)]
            max_color = [int(self.target_lab[0] + 50 * threshold * 2),
                         int(self.target_lab[1] + 50 * threshold),
                         int(self.target_lab[2] + 50 * threshold)]
            target_color = self.target_lab, min_color, max_color
            lowerb = tuple(target_color[1])
            upperb = tuple(target_color[2])
        else:
            lowerb = tuple(color['min'])
            upperb = tuple(color['max'])
        for roi in self.rois:
            blob = image[int(roi[0]*h):int(roi[1]*h), int(roi[2]*w):int(roi[3]*w)]  # roi(intercept roi)
            img_lab = cv2.cvtColor(blob, cv2.COLOR_RGB2LAB)  # rgblab(convert rgb into lab)
            img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)  # (perform Gaussian filtering to reduce noise)
            # mask = cv2.inRange(img_blur, tuple(target_color[1]), tuple(target_color[2]))  # (image binarization)
            mask = cv2.inRange(img_blur, lowerb, upperb)  # (image binarization)
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # (corrode)
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # (dilate)
            # cv2.imshow('section:{}:{}'.format(roi[0], roi[1]), cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]  # (find the contour)
            max_contour_area = self.get_area_max_contour(contours, 30)  # (get the contour corresponding to the largest contour)
            if max_contour_area is not None:
                hit_count += 1
                rect = cv2.minAreaRect(max_contour_area[0])  # (minimum circumscribed rectangle)
                box = np.intp(cv2.boxPoints(rect))  # (four corners)
                for j in range(4):
                    box[j, 1] = box[j, 1] + int(roi[0]*h)
                cv2.drawContours(result_image, [box], -1, (0, 255, 255), 2)  # (draw the rectangle composed of four points)

                # (acquire the diagonal points of the rectangle)
                pt1_x, pt1_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                # (center point of the line)
                line_center_x, line_center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2

                cv2.circle(result_image, (int(line_center_x), int(line_center_y)), 5, (0, 0, 255), -1)   # (draw the center point)
                centroid_sum += line_center_x * roi[-1]
        # Require at least two ROI hits to consider the line valid; otherwise treat as no detection.
        if centroid_sum == 0 or hit_count < 2:
            return result_image, None, hit_count
        center_pos = centroid_sum / self.weight_sum  # (calculate the center point according to the ratio)
        deflection_angle = -math.atan((center_pos - (w / 2.0)) / (h / 2.0))   # (calculate the line angle)
        return result_image, deflection_angle, hit_count

class LineFollowingNode(Node):
    """ROS node that handles the full game: vision, lidar, and drive commands."""

    def __init__(self, name):
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        self.name = name
        self.color = ''
        # self.camera_type = 'Stereo'
        self.set_callback = False
        self.is_running = False
        self.color_picker = None
        self.follower = None
        self.scan_angle = math.radians(45)
        self.scan_angle = math.radians(45)
        self.pid = pid.PID(0.030, 0.003, 0.0) # Edit this line to change PID values original values (0.005, 0.001, 0.)
        self.empty = 0
        self.count = 0
        self.stop = False
        self.threshold = 0.5
        self.stop_threshold = 0.4
        self.scatter_block = False
        self.lock = threading.RLock()
        self.image_sub = None
        self.lidar_sub = None
        self.image_height = None
        self.image_width = None
        self.bridge = CvBridge()
        self.use_color_picker = True
        self.lab_data = _load_lab_config()
        self.image_queue = queue.Queue(2)
        self.camera_type = _get_camera_type()
        # Select lab profile: prefer configured camera, else ascamera, else first available, else None.
        lab_map = self.lab_data.get('lab', {})
        if self.camera_type in lab_map:
            self.lab_lookup_type = self.camera_type
        elif 'ascamera' in lab_map:
            self.lab_lookup_type = 'ascamera'
        elif lab_map:
            self.lab_lookup_type = next(iter(lab_map))
        else:
            self.lab_lookup_type = None
        self.lidar_type = os.environ.get('LIDAR_TYPE')
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.missing_profile_logged = False
        self._servo_log_cooldown = 0.50  # seconds between logging servo commands
        self._last_servo_log = 0.0
        # Camera topic and frame counters for debugging visibility.
        self.camera_topic = '/ascamera/camera_publisher/rgb0/image'
        self.image_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.frame_count = 0
        self.last_frame_time = None
        self.debug_log_every_n = self.declare_parameter('debug_log_every_n', 30).value
        self.pwm_pub = self.create_publisher(SetPWMServoState,'ros_robot_controller/pwm_servo/set_state',10)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)  # (chassis control)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)  # (publish the image processing result)
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)  # (enter the game)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)  # (exit the game)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)  # (start the game)
        self.create_service(SetString, '~/set_color', self.set_color_srv_callback)
        self.create_service(SetPoint, '~/set_target_color', self.set_target_color_srv_callback)  # (set the color)
        self.create_service(Trigger, '~/get_target_color', self.get_target_color_srv_callback)   # (get the color)
        self.create_service(SetFloat64, '~/set_threshold', self.set_threshold_srv_callback)  # (set the threshold)
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self._last_stop_state = False  # track lidar stop toggles for debug

        Heart(self, self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # (heartbeat package)
        # Default debug to True so the UI window opens when not set via parameters
        self.declare_parameter('debug', True)
        self.debug = self.get_parameter('debug').value
        if self.debug: 
            threading.Thread(target=self.main, daemon=True).start()
        else:
            self.get_logger().info("Debug UI disabled (no color picker window). Set param 'debug' true to enable.")
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        if self.debug:
            self.get_logger().info(f"Debug enabled; subscribing to camera: {self.camera_topic}")
            if self.lab_lookup_type and self.lab_lookup_type != self.camera_type:
                self.get_logger().info(f"Using lab profile '{self.lab_lookup_type}' as fallback for color ranges")
        
    def pwm_controller(self,position_data):
        pwm_list = []
        msg = SetPWMServoState()
        msg.duration = 0.2
        for i in range(len(position_data)):
            pos = PWMServoState()
            pos.id = [i+1]
            pos.position = [int(position_data[i])]
            pwm_list.append(pos)
        msg.state = pwm_list
        self.pwm_pub.publish(msg)

    def get_node_state(self, request, response):
        response.success = True
        return response

    def main(self):
        try:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        except Exception as exc:
            self.get_logger().warning(f"Failed to create debug window: {exc}")
            return
        while True:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if self.debug and not self.set_callback:
                self.set_callback = True
                # (set a callback function for mouse click event)
                cv2.setMouseCallback("result", self.mouse_callback)
            k = cv2.waitKey(1)
            if k != -1:
                break
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.get_logger().info("x:{} y{}".format(x, y))
            msg = SetPoint.Request()
            if self.image_height is not None and self.image_width is not None:
                msg.data.x = x / self.image_width
                msg.data.y = y / self.image_height
                self.set_target_color_srv_callback(msg, SetPoint.Response())

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "line following enter")
        if self.camera_type != 'ascamera':
            self.pwm_controller([1850,1500]) ## Pan / tilt originally [1850, 1500]
        with self.lock:
            self.stop = False
            self.is_running = False  # will be re-enabled after color is chosen
            self.color_picker = None
            self.pid = pid.PID(1.1, 0.0, 0.0)
            self.follower = None
            self.threshold = 0.5
            #self.camera_type = os.environ['DEPTH_CAMERA_TYPE']
            self.empty = 0
            if self.image_sub is None:
                self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, qos_profile=self.image_qos)  # subscribe to the camera
                if self.debug:
                    self.get_logger().info("Subscribed to camera: %s" % self.camera_topic)
            if self.lidar_sub is None:
                qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
                self.lidar_sub = self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, qos)  # (subscribe to Lidar data)
                self._set_servo_position(((10, 200), (5, 500), (4, 90), (3, 150), (2, 645), (1, 500)), 1, "line_follow_pose") # Use this to edit the arm position, parameters start from the gripper and end at the arm base
            self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "line following exit")
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
            self.color_picker = None
            self.pid = pid.PID(0.00, 0.001, 0.0)
            self.follower = None
            self.threshold = 0.5
            self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "exit"
        return response

    def set_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_target_color")
        with self.lock:
            self.use_color_picker = True
            x, y = request.data.x, request.data.y
            self.follower = None
            # Enable motion once a target is selected.
            self.is_running = True
            if x == -1 and y == -1:
                self.color_picker = None
            else:
                self.color_picker = ColorPicker(request.data, 5)
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_target_color"
        return response

    def get_target_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "get_target_color")
        response.success = False
        response.message = "get_target_color"
        with self.lock:
            if self.follower is not None:
                response.success = True
                rgb = self.follower.target_rgb
                response.message = "{},{},{}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.is_running = request.data
            self.empty = 0
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

    def _set_servo_position(self, positions, duration=1.0, label=None):
        """Send a servo command and log it at most every 500 ms to reduce spam."""
        now = time.time()
        if (now - self._last_servo_log) >= self._servo_log_cooldown:
            tag = f" {label}" if label else ""
            self.get_logger().info(f"[servo]{tag}: {positions}")
            self._last_servo_log = now
        set_servo_position(self.joints_pub, duration, positions)

    def set_color_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % 'set_color')
        with self.lock:
            self.color = request.data
            self.use_color_picker = False
            self.missing_profile_logged = False
            # Enable motion when a color is explicitly set.
            self.is_running = True
        response.success = True
        response.message = "set_color"
        return response

    def lidar_callback(self, lidar_data):
        # (data size= scanning angle/ the increased angle per scan)
        if self.lidar_type != 'G4':
            min_index = int(math.radians(MAX_SCAN_ANGLE / 2.0) / lidar_data.angle_increment)
            max_index = int(math.radians(MAX_SCAN_ANGLE / 2.0) / lidar_data.angle_increment)
            left_ranges = lidar_data.ranges[:max_index]  # (left data)
            right_ranges = lidar_data.ranges[::-1][:max_index]  # (right data)
        elif self.lidar_type == 'G4':
            min_index = int(math.radians((360 - MAX_SCAN_ANGLE) / 2.0) / lidar_data.angle_increment)
            max_index = int(math.radians(180) / lidar_data.angle_increment)
            left_ranges = lidar_data.ranges[min_index:max_index][::-1]  # (the left data)
            right_ranges = lidar_data.ranges[::-1][min_index:max_index][::-1]  #  (the right data)

        # (Get data according to settings)
        angle = self.scan_angle / 2
        angle_index = int(angle / lidar_data.angle_increment + 0.50)
        left_range, right_range = np.array(left_ranges[:angle_index]), np.array(right_ranges[:angle_index])

        left_nonzero = left_range.nonzero()
        right_nonzero = right_range.nonzero()
        left_nonan = np.isfinite(left_range[left_nonzero])
        right_nonan = np.isfinite(right_range[right_nonzero])
        # (Take the nearest distance left and right)
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
        # Emit a debug log when stop state toggles so we know if lidar is blocking motion.
        if self.debug and self.stop != self._last_stop_state:
            state = "STOPPED" if self.stop else "CLEAR"
            self.get_logger().info(f"[lidar] stop={state}, min_left={min_dist_left_[:1] if len(min_dist_left_)>0 else 'n/a'}, min_right={min_dist_right_[:1] if len(min_dist_right_)>0 else 'n/a'}")
        self._last_stop_state = self.stop

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        self.image_height, self.image_width = rgb_image.shape[:2]
        result_image = np.copy(rgb_image)  #  (the image used to display the result)
        self.frame_count += 1
        self.last_frame_time = self.get_clock().now()
        if self.debug and self.frame_count % max(1, int(self.debug_log_every_n)) == 0:
            self.get_logger().info(f"[debug] frame {self.frame_count} size=({self.image_width}x{self.image_height}) from {self.camera_topic}")
        with self.lock:
            if self.use_color_picker:
                # (color picker and line recognition are exclusive. If there is color picker, start picking)
                if self.color_picker is not None:  # (color picker exists)
                    try:
                        target_color, result_image = self.color_picker(rgb_image, result_image)
                        if target_color is not None:
                            self.color_picker = None
                            self.follower = LineFollower(target_color, self)
                            self.get_logger().info("target color: {}".format(target_color))
                    except Exception as e:
                        self.get_logger().error(str(e))
                else:
                    twist = Twist()
                    twist.linear.x = 0.05 # Speed control variable
                    if self.follower is not None:
                        try:
                            result_image, deflection_angle, hit_count = self.follower(rgb_image, result_image, self.threshold)
                            # Do not block motion on scatter anymore; only lidar stop will block.
                            self.scatter_block = False
                            effective_stop = self.stop
                            if deflection_angle is not None and self.is_running and not effective_stop:
                                self.pid.update(deflection_angle)
                                if 'Acker' in self.machine_type:
                                    steering_angle = common.set_range(-self.pid.output, -math.radians(40), math.radians(40))
                                    if steering_angle != 0:
                                        R = 0.145/math.tan(steering_angle)
                                        twist.angular.z = twist.linear.x/R
                                else:
                                    twist.angular.z = common.set_range(-self.pid.output, -1.0, 1.0)
                                self.mecanum_pub.publish(twist)
                                if self.debug and self.frame_count % max(1, int(self.debug_log_every_n)) == 0:
                                    self.get_logger().info(
                                        f"[motion] angle={deflection_angle:.4f}, lin_x={twist.linear.x:.3f}, ang_z={twist.angular.z:.3f}, stop={effective_stop}, hits={hit_count}"
                                    )
                            elif effective_stop:
                                if self.debug:
                                    reason = "lidar" if self.stop else "no_roi"
                                    self.get_logger().debug(f"Motion suppressed: stop flag set ({reason}), hits={hit_count}")
                                self.mecanum_pub.publish(Twist())
                            else:
                                self.pid.clear()
                                if self.debug:
                                    self.get_logger().debug(
                                        f"Suppressing motion (angle={deflection_angle}, hits={hit_count}, is_running={self.is_running}, stop={effective_stop})"
                                    )
                        except Exception as e:
                            self.get_logger().error(str(e))
            else:
                twist = Twist()
                lab_profiles = self.lab_data.get('lab', {})
                color_profile = None
                if (
                    self.color in common.range_rgb
                    and self.lab_lookup_type
                    and self.lab_lookup_type in lab_profiles
                ):
                    color_profile = lab_profiles[self.lab_lookup_type].get(self.color)
                if color_profile:
                    twist.linear.x = 0.05  # Speed control variable
                    self.follower = LineFollower([None, common.range_rgb[self.color]], self)
                    result_image, deflection_angle, hit_count = self.follower(
                        rgb_image,
                        result_image,
                        self.threshold,
                        color_profile,
                        False,
                    )
                    # Do not block motion on scatter anymore; only lidar stop will block.
                    self.scatter_block = False
                    effective_stop = self.stop
                    if deflection_angle is not None and self.is_running and not effective_stop:
                        self.pid.update(deflection_angle)
                        if 'Acker' in self.machine_type:
                            steering_angle = common.set_range(-self.pid.output, -math.radians(40), math.radians(40))
                            if steering_angle != 0:
                                R = 0.145/math.tan(steering_angle)
                                twist.angular.z = twist.linear.x/R
                        else:
                            twist.angular.z = common.set_range(-self.pid.output, -1.0, 1.0)
                        self.mecanum_pub.publish(twist)
                        if self.debug and self.frame_count % max(1, int(self.debug_log_every_n)) == 0:
                            self.get_logger().info(
                                f"[motion] angle={deflection_angle:.4f}, lin_x={twist.linear.x:.3f}, ang_z={twist.angular.z:.3f}, stop={effective_stop}, hits={hit_count}"
                            )
                    elif self.stop:
                        if self.debug:
                            self.get_logger().debug("Motion suppressed: stop flag set")
                        self.mecanum_pub.publish(Twist())
                    else:
                        self.pid.clear()
                        if self.debug:
                            self.get_logger().debug(
                                f"Suppressing motion (angle={deflection_angle}, hits={hit_count}, is_running={self.is_running}, stop={effective_stop})"
                            )
                else:
                    if self.debug and not self.missing_profile_logged:
                        self.get_logger().warning(
                            f"Missing lab profile for camera '{self.camera_type}' (lookup '{self.lab_lookup_type}') color '{self.color}'"
                        )
                        self.missing_profile_logged = True
                    self.mecanum_pub.publish(twist)
        if self.debug:
            if self.image_queue.full():
                # (if the queue is full, remove the oldest image)
                self.image_queue.get()
                # (put the image into the queue)
            self.image_queue.put(result_image)
        # Always publish the processed image so tools like rqt_image_view can see it even when debug UI is on.
        self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))

def main():
    if not rclpy.ok():
        rclpy.init()
    node = LineFollowingNode('line_following')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()









