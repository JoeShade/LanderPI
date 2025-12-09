#!/usr/bin/env python3
"""
Orchestrate the full-featured line following, green beacon navigation, and HRI
programs in sequence without user intervention.

This runner does **not** re-implement the behaviors from ``line_following.py``,
``green_nav.py``, or ``HRI.py``. Instead, it launches those programs as child
processes so every feature (voice feedback, ROI selection, lidar handling,
servo gestures, etc.) remains available exactly as in the original scripts.

Flow:
1. Start ``line_following.py`` with the built-in black line profile and follow
   the line until it disappears for several consecutive frames.
2. Switch to ``green_nav.py`` to search for the green beacon while performing
   lidar-based avoidance. The runner monitors camera images to determine when
   the beacon fills enough of the image.
3. When the beacon is considered reached, stop green navigation and launch
   ``HRI.py`` to enable gesture-based interaction.

The runner watches camera images for transitions while letting the stage nodes
drive the robot, so the original behaviors stay intact and the scenario can run
hands-free.
"""

import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from std_srvs.srv import SetBool
from interfaces.srv import SetString
from scenario_pkg.roi_config import ROI_TABLE, get_rois

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# LAB bounds for the default black line (hard-coded so no color picking is
# required).
BLACK_LAB_RANGE = {"min": [0, 0, 0], "max": [40, 120, 120]}
GREEN_LAB_RANGE = {"min": [0, 80, 0], "max": [255, 120, 120]}  # broad green

# How many frames without a line before we transition away from line following.
LINE_LOST_FRAMES = 10

# How large the green beacon should appear (fraction of the image area) before
# handing control to HRI.
BEACON_AREA_THRESHOLD = 0.06

# Debugging defaults.
DEBUG_LOG_EVERY_N_FRAMES = 15
DEBUG_OUTPUT_DIR = Path("/tmp/scenario_debug")


def _resolve_camera_topic(camera_type: str) -> str:
    """Pick a camera topic that matches the active sensor; env IMAGE_TOPIC overrides."""
    env_topic = os.environ.get("IMAGE_TOPIC") or os.environ.get("CAMERA_TOPIC")
    if env_topic:
        return env_topic
    if camera_type == "usb_cam":
        return "/camera/image"
    if camera_type in ("ascamera", "aurora"):
        return "/ascamera/camera_publisher/rgb0/image"
    return "/camera/image_raw"


class Stage(Enum):
    """High-level mission stages."""

    LINE = 1
    GREEN = 2
    HRI = 3


class LineWatcher:
    """Utility that mirrors the ROI logic from line_following.py for continuity."""

    def __init__(self, rois: Tuple[Tuple[float, float, float, float, float], ...]):
        self.rois = rois
        self.weight_sum = sum(roi[-1] for roi in rois)
        # Require more substantive detections so scattered carpet specks are ignored.
        # The ratios are slightly conservative so that once the line is gone we
        # quickly classify frames as misses instead of bouncing on tiny blobs.
        self.min_contour_area = 500
        self.min_mask_ratio = 0.01  # fraction of ROI pixels that must be non-zero
        self.min_total_mask_ratio = 0.03  # combined ROI coverage required to count as "line seen"
        self.min_hit_rois = 1

    @staticmethod
    def _largest_contour(contours, threshold=120):
        contour_area = zip(contours, tuple(map(lambda c: abs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if contour_area:
            return max(contour_area, key=lambda c_a: c_a[1])
        return None

    def detect_angle(self, image: np.ndarray, lowerb, upperb) -> Tuple[Optional[float], int, float]:
        """Return (steering angle, hit_count, total_mask_ratio). Angle is None if too few ROI hits."""

        h, w = image.shape[:2]
        centroid_sum = 0.0
        hit_count = 0
        total_mask_ratio = 0.0
        for roi in self.rois:
            blob = image[int(roi[0] * h): int(roi[1] * h), int(roi[2] * w): int(roi[3] * w)]
            mask = cv2.inRange(cv2.cvtColor(blob, cv2.COLOR_RGB2LAB), lowerb, upperb)
            # Skip ROIs where almost nothing is detected.
            mask_ratio = float(cv2.countNonZero(mask)) / max(mask.size, 1)
            if mask_ratio < self.min_mask_ratio:
                continue
            total_mask_ratio += mask_ratio
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            max_contour_area = self._largest_contour(contours, self.min_contour_area)
            if max_contour_area is not None:
                hit_count += 1
                rect = cv2.minAreaRect(max_contour_area[0])
                box = np.intp(cv2.boxPoints(rect))
                center_x = (box[0, 0] + box[2, 0]) / 2
                center_y = (box[0, 1] + box[2, 1]) / 2
                cv2.circle(blob, (int(center_x), int(center_y)), 3, (255, 0, 0), -1)
                centroid_sum += center_x * roi[-1]

        if centroid_sum == 0 or hit_count < self.min_hit_rois or total_mask_ratio < self.min_total_mask_ratio:
            return None, hit_count, total_mask_ratio
        center_pos = centroid_sum / max(self.weight_sum, 1e-6)
        deflection_angle = -math.atan((center_pos - (w / 2.0)) / (h / 2.0))
        return deflection_angle, hit_count, total_mask_ratio


class GreenWatcher:
    """Simplified green blob monitor used only for transition detection."""

    @staticmethod
    def area_ratio(image: np.ndarray, lowerb, upperb) -> float:
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (5, 5), 3)
        mask = cv2.inRange(img_blur, lowerb, upperb)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / max(float(image.shape[0] * image.shape[1]), 1e-6)
        return area_ratio


class ScenarioRunner(Node):
    """ROS 2 node that launches and supervises the full-featured stage nodes."""

    def __init__(self):
        # ``rclpy`` must be initialized exactly once per process. Launch files or
        # other entrypoints could have already called ``rclpy.init()``, so guard
        # the call to avoid a runtime crash from double-initialization.
        if not rclpy.ok():
            rclpy.init()
        super().__init__("scenario_runner")

        # Debug toggles can be adjusted via ROS parameters or the command line
        # (e.g., ros2 run ... --ros-args -p debug_mode:=true -p save_debug_images:=true).
        self.debug_mode: bool = self.declare_parameter("debug_mode", False).value
        self.save_debug_images: bool = self.declare_parameter("save_debug_images", False).value
        debug_dir_param = self.declare_parameter("debug_output_dir", str(DEBUG_OUTPUT_DIR)).value
        self.debug_output_dir = Path(debug_dir_param)
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_every_n = max(1, int(self.declare_parameter("debug_log_every_n", DEBUG_LOG_EVERY_N_FRAMES).value))

        # Camera helpers for transition sensing.
        self.bridge = CvBridge()
        # Camera subscription (align with line_following/green_nav expectations).
        camera_type = os.environ.get("DEPTH_CAMERA_TYPE", "aurora")
        self.camera_topic = _resolve_camera_topic(camera_type)
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self._image_cb, qos)

        # Resolve the installed share dir so we can launch the original scripts even from install space.
        self.share_dir = Path(get_package_share_directory("scenario_pkg"))

        # Stage tracking.
        self.stage = Stage.LINE
        self.current_process: Optional[subprocess.Popen] = None
        self.line_lost_frames = 0
        self.line_transition_armed = True  # start counting immediately to catch missing lines
        # Two counters: raw frames received and frames we actually inspect.
        self.raw_frame_counter = 0  # counts every incoming image
        self.frame_counter = 0  # counts processed images
        self.process_every_n = max(1, int(self.declare_parameter("runner_process_every_n", 5).value))
        self.last_image: Optional[np.ndarray] = None
        self.child_log_threads: Tuple[threading.Thread, ...] = tuple()
        self.stage_start_time: float = time.time()
        self.mission_complete = False
        self.shutdown_requested = False
        # Stall timer arms only after motion starts (which requires a selected color).
        self.last_motion_time: Optional[float] = None
        self.last_odom_motion_time: Optional[float] = None
        self.motion_timeout = 3.0
        self.odom_moving_threshold = 0.02  # m/s considered "moving"
        self.motion_check_period = 0.5
        self.motion_timeout_triggered = False
        self.last_cmd_vel_log_time = 0.0
        self.last_odom_log_time = 0.0
        self.motion_log_interval = 2.0
        self.odom_seen = False
        self.last_odom_msg_time: Optional[float] = None
        self.missing_odom_logged = False
        self.transition_lock = threading.Lock()
        camera_type = os.environ.get("DEPTH_CAMERA_TYPE", "aurora")
        self.line_watcher = LineWatcher(get_rois(camera_type))
        self.green_watcher = GreenWatcher()

        # Precompute LAB bounds used for transition detection.
        self.black_lower = tuple(BLACK_LAB_RANGE["min"])
        self.black_upper = tuple(BLACK_LAB_RANGE["max"])
        self.green_lower = tuple(GREEN_LAB_RANGE["min"])
        self.green_upper = tuple(GREEN_LAB_RANGE["max"])

        # Allow external introspection when diagnosing issues.
        self.create_service(Trigger, "get_status", self._handle_get_status)
        # Allow manual override when recovering from stalls or testing.
        self.create_service(SetString, "set_stage", self._handle_set_stage)
        # Safety stop publisher in case a child process is terminated mid-motion.
        self.cmd_vel_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        # Track outgoing motion commands to detect stalls.
        qos_cmd = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.cmd_vel_sub = self.create_subscription(Twist, "/controller/cmd_vel", self._cmd_vel_cb, qos_cmd)
        # Track odometry to confirm the robot is actually moving.
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, qos_cmd)

        # Start the first stage immediately.
        self._launch_stage(Stage.LINE)

        # Periodic heartbeat to log what is active.
        self.create_timer(5.0, self._log_status)
        # Motion watchdog so stalls trigger even if images are not processed.
        self.create_timer(self.motion_check_period, self._check_motion_timeout)

    # ------------------------------------------------------------------
    # Transition handling
    # ------------------------------------------------------------------
    def _launch_stage(self, stage: Stage):
        """Start the requested stage's original script as a subprocess."""

        with self.transition_lock:
            self._stop_child_process()
            self._launch_stage_locked(stage)

    def _launch_stage_locked(self, stage: Stage):
        """Internal helper; caller must hold transition_lock."""
        # Give the OS a moment to release resources (audio, camera) from the previous process.
        time.sleep(1.0)

        script_map = {
            Stage.LINE: str(self.share_dir / "line_following.py"),
            Stage.GREEN: str(self.share_dir / "green_nav.py"),
            Stage.HRI: str(self.share_dir / "HRI.py"),
        }

        script = script_map[stage]
        self.get_logger().info(f"Launching stage {stage.name}: {script}")
        env = os.environ.copy()
        env.setdefault("DEPTH_CAMERA_TYPE", "aurora")
        env.setdefault("VOICE_VOLUME", "90")
        env.setdefault("LINE_COLOR", "black")
        # Ensure the line follower starts without user color picking.
        env.setdefault("LINE_USE_DEFAULT", "1")

        stdout_pipe = subprocess.PIPE if self.debug_mode else None
        stderr_pipe = subprocess.PIPE if self.debug_mode else None
        # Use unbuffered output so logs stream to ros2 log.
        try:
            self.current_process = subprocess.Popen(
                [sys.executable, "-u", script], env=env, stdout=stdout_pipe, stderr=stderr_pipe, text=True, bufsize=1
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Failed to launch stage {stage.name} (script: {script}): {exc}. Stopping runner to avoid orphaned state."
            )
            self.shutdown_requested = True
            self.mission_complete = True
            try:
                self.cmd_vel_pub.publish(Twist())
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass
            return
        self.stage = stage
        self.line_lost_frames = 0
        self.frame_counter = 0
        self.stage_start_time = time.time()
        self.motion_timeout_triggered = False
        if stage == Stage.LINE:
            self.last_motion_time = None
            self.last_odom_motion_time = None
        elif stage == Stage.GREEN:
            # Arm timeout immediately for green_nav to catch stalls.
            self.last_motion_time = time.time()
            self.last_odom_motion_time = None
        else:
            # HRI should not have a motion timeout.
            self.last_motion_time = None
            self.last_odom_motion_time = None
        self.odom_seen = False
        self.last_odom_msg_time = None
        self.missing_odom_logged = False
        self.get_logger().info(f"Stage {self.stage.name} launched (pid={self.current_process.pid if self.current_process else 'n/a'})")
        self._attach_child_loggers()

        # Configure the node after a short delay to let services come up.
        if stage == Stage.LINE:
            threading.Thread(target=self._configure_line_node, daemon=True).start()
        elif stage == Stage.GREEN:
            threading.Thread(target=self._configure_green_node, daemon=True).start()

        if self.save_debug_images:
            self._save_debug_image(self.last_image, f"launch_{stage.name.lower()}")

    def _stop_child_process(self):
        if self.current_process and self.current_process.poll() is None:
            self.get_logger().info("Stopping previous stage before launching new one")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
                self.get_logger().info(f"Child exited with code {self.current_process.returncode}")
            except subprocess.TimeoutExpired:
                self.get_logger().warning("Force-killing previous stage")
                self.current_process.kill()
                try:
                    self.current_process.wait(timeout=3)
                    self.get_logger().info(f"Child killed; exit code {self.current_process.returncode}")
                except subprocess.TimeoutExpired:
                    self.get_logger().error("Child did not terminate after kill()")
        # SAFETY: force an immediate stop to clear any lingering wheel commands.
        try:
            self.cmd_vel_pub.publish(Twist())
        except Exception:
            pass
        self._join_child_loggers()
        self.current_process = None

    def _call_service(self, srv_type, name: str, request):
        """Utility to synchronously call a service with retry."""

        client = self.create_client(srv_type, name)
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warning(f"Service {name} not available")
            return False
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda fut: done.set())
        if not done.wait(timeout=8.0):
            self.get_logger().warning(f"Service {name} timed out")
            return False
        if future.exception() is not None:
            self.get_logger().warning(f"Service {name} failed: {future.exception()}")
            return False
        success = getattr(future.result(), "success", True)
        self.get_logger().info(f"Service {name} completed (success={success})")
        return success

    def _attach_child_loggers(self):
        if not self.debug_mode or not self.current_process:
            return
        threads = []
        if self.current_process.stdout:
            threads.append(threading.Thread(target=self._pipe_logger, args=(self.current_process.stdout, "stdout"), daemon=True))
        if self.current_process.stderr:
            threads.append(threading.Thread(target=self._pipe_logger, args=(self.current_process.stderr, "stderr"), daemon=True))
        for t in threads:
            t.start()
        self.child_log_threads = tuple(threads)

    def _pipe_logger(self, pipe, label: str):
        for line in iter(pipe.readline, ""):
            clean = line.rstrip() if line else ""
            if clean:
                self.get_logger().info(f"[{self.stage.name} {label}] {clean}")
        pipe.close()

    def _join_child_loggers(self):
        for t in self.child_log_threads:
            t.join(timeout=0.5)
        self.child_log_threads = tuple()

    def _configure_line_node(self):
        """Set the color to black and start the line follower."""

        # Wait briefly so the child can advertise services.
        self.get_clock().sleep_for(Duration(seconds=1))
        base_name = "/line_following"
        self._call_service(SetBool, f"{base_name}/set_running", SetBool.Request(data=True))
        self._call_service(Trigger, f"{base_name}/enter", Trigger.Request())

    def _configure_green_node(self):
        """Start the green beacon navigator with its default parameters."""

        self.get_clock().sleep_for(Duration(seconds=1))
        base_name = "/green_nav"
        self._call_service(Trigger, f"{base_name}/enter", Trigger.Request())

    def _handle_get_status(self, request, response):  # noqa: ARG002
        running = self.current_process and self.current_process.poll() is None
        response.success = True
        response.message = (
            f"stage={self.stage.name}, child_running={running}, "
            f"line_lost_frames={self.line_lost_frames}, frame_counter={self.frame_counter}"
        )
        return response

    def _handle_set_stage(self, request, response):
        """Manual override to force a stage transition (e.g., to recover from stalls)."""
        target = (request.data or "").strip().upper()
        next_stage = None
        if target in ("LINE", "GREEN", "HRI"):
            next_stage = Stage[target]
        elif target == "NEXT":
            next_stage = Stage.GREEN if self.stage == Stage.LINE else Stage.HRI
        else:
            response.success = False
            response.message = "Invalid stage. Use LINE, GREEN, HRI, or NEXT."
            return response

        self.get_logger().info(f"Manual override: forcing stage {next_stage.name}")
        self._launch_stage(next_stage)
        response.success = True
        response.message = f"Switched to {next_stage.name}"
        return response

    # ------------------------------------------------------------------
    # Image callbacks used purely for transition decisions
    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image):
        # Throttle processing to reduce double-decoding load; the child node keeps full rate.
        self.raw_frame_counter += 1
        if self.raw_frame_counter % self.process_every_n != 0:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"Failed to decode image: {exc}")
            return

        self.last_image = image
        self.frame_counter += 1
        if self.stage == Stage.LINE:
            self._handle_line_stage(image)
        elif self.stage == Stage.GREEN:
            self._handle_green_stage(image)
        if self.frame_counter % self.debug_log_every_n == 0:
            self.get_logger().info(
                f"[frames] raw={self.raw_frame_counter}, processed={self.frame_counter}, stage={self.stage.name}, process_every_n={self.process_every_n}"
            )

    def _handle_line_stage(self, image: np.ndarray):
        angle, hits, coverage = self.line_watcher.detect_angle(image, self.black_lower, self.black_upper)
        line_seen = angle is not None
        if line_seen:
            self.line_lost_frames = 0
        elif self.line_transition_armed:
            self.line_lost_frames += 1

        if self.debug_mode and self.frame_counter % self.debug_log_every_n == 0:
            self.get_logger().info(
                f"[LINE debug] frame={self.frame_counter}, angle={angle if angle is not None else 'none'}, "
                f"hits={hits}, coverage={coverage:.4f}, lost_frames={self.line_lost_frames}"
            )

        if self.line_transition_armed and self.line_lost_frames >= LINE_LOST_FRAMES:
            self.get_logger().info("Line lost for %d frames, switching to GREEN stage" % self.line_lost_frames)
            if self.save_debug_images:
                self._save_debug_image(image, "line_lost")
            self._launch_stage(Stage.GREEN)
            return

        # Secondary fallback: if no motion command for >motion_timeout seconds during LINE, switch to GREEN.
        if (
            self.stage == Stage.LINE
            and self.last_motion_time is not None
            and not self.motion_timeout_triggered
            and (time.time() - self.last_motion_time) > self.motion_timeout
        ):
            self.motion_timeout_triggered = True
            self.get_logger().info(
                f"No motion command for {self.motion_timeout} seconds; switching to GREEN stage as fallback (stage_time={time.time() - self.stage_start_time:.1f}s)."
            )
            if self.save_debug_images:
                self._save_debug_image(image, "motion_timeout")
            self._launch_stage(Stage.GREEN)

    def _handle_green_stage(self, image: np.ndarray):
        # Ignore green-stage detections until the camera/servos have time to move.
        if time.time() - self.stage_start_time < 2.0:
            return
        area_ratio = self.green_watcher.area_ratio(image, self.green_lower, self.green_upper)
        if self.debug_mode and self.frame_counter % self.debug_log_every_n == 0:
            self.get_logger().info(f"[GREEN debug] frame={self.frame_counter}, beacon_area={area_ratio:.4f}")
        if area_ratio >= BEACON_AREA_THRESHOLD:
            self.get_logger().info(
                f"Beacon reached (area ratio {area_ratio:.3f} >= {BEACON_AREA_THRESHOLD}), switching to HRI (stage_time={time.time() - self.stage_start_time:.1f}s)"
            )
            if self.save_debug_images:
                self._save_debug_image(image, "beacon_reached")
            self._launch_stage(Stage.HRI)
            return

        # If no motion command for >motion_timeout seconds during GREEN, switch to HRI as fallback.
        if (
            self.stage == Stage.GREEN
            and self.last_motion_time is not None
            and not self.motion_timeout_triggered
            and (time.time() - self.last_motion_time) > self.motion_timeout
        ):
            self.motion_timeout_triggered = True
            self.get_logger().info(
                f"No motion command for {self.motion_timeout} seconds during GREEN; switching to HRI stage as fallback (stage_time={time.time() - self.stage_start_time:.1f}s)."
            )
            if self.save_debug_images:
                self._save_debug_image(image, "green_motion_timeout")
            self._launch_stage(Stage.HRI)

    def _check_motion_timeout(self):
        """Detect stalls when Twist commands persist but odom shows no motion."""
        if self.stage == Stage.HRI or self.motion_timeout_triggered:
            return
        now = time.time()
        recent_command = self.last_motion_time is not None and (now - self.last_motion_time) < self.motion_timeout
        odom_stalled = self.last_odom_motion_time is None or (now - self.last_odom_motion_time) > self.motion_timeout
        if not self.odom_seen:
            if not self.missing_odom_logged:
                self.get_logger().warning("Stall watchdog armed but no /odom received yet; skipping stall check until odom arrives.")
                self.missing_odom_logged = True
            return
        if self.last_odom_msg_time and (now - self.last_odom_msg_time) > max(self.motion_timeout, 2 * self.motion_check_period):
            self.get_logger().warning(f"Odom data stale ({now - self.last_odom_msg_time:.2f}s); skipping stall check this cycle.")
            return
        self.get_logger().info(
            f"[motion_watch] stage={self.stage.name}, recent_cmd={recent_command}, "
            f"since_cmd={(now - self.last_motion_time) if self.last_motion_time else 'n/a'}, "
            f"since_odom={(now - self.last_odom_motion_time) if self.last_odom_motion_time else 'n/a'}"
        )
        if self.stage in (Stage.LINE, Stage.GREEN) and recent_command and odom_stalled:
            self.motion_timeout_triggered = True
            next_stage = Stage.GREEN if self.stage == Stage.LINE else Stage.HRI
            self.get_logger().info(
                f"Motion stall detected: commands active but odom idle for {self.motion_timeout} seconds; switching to {next_stage.name}."
            )
            if self.save_debug_images:
                self._save_debug_image(self.last_image, f"stall_{self.stage.name.lower()}")
            self._launch_stage(next_stage)

    # ------------------------------------------------------------------
    # Logging and shutdown
    # ------------------------------------------------------------------
    def _log_status(self):
        running = self.current_process and self.current_process.poll() is None
        pid_info = f"pid={self.current_process.pid}" if self.current_process else "pid=n/a"
        retcode = self.current_process.returncode if self.current_process else None
        self.get_logger().info(
            f"Stage={self.stage.name}, child_running={running}, {pid_info}, returncode={retcode}, lost_frames={self.line_lost_frames}, "
            f"mission_complete={self.mission_complete}, shutdown_requested={self.shutdown_requested}, motion_timeout_triggered={self.motion_timeout_triggered}"
        )
        # If HRI exits cleanly, treat the mission as complete.
        if self.stage == Stage.HRI and not running and not self.mission_complete:
            self.mission_complete = True
            self.shutdown_requested = True
            self.get_logger().info("Mission complete: HRI exited cleanly; shutting down runner.")
            try:
                self.cmd_vel_pub.publish(Twist())
            except Exception:
                pass
            # Graceful shutdown of rclpy; node cleanup happens in main thread after spin exits.
            try:
                rclpy.shutdown()
            except Exception:
                pass
        elif not running and self.current_process:
            self.get_logger().warning(f"Child for stage {self.stage.name} exited unexpectedly (returncode={self.current_process.returncode})")

    def _save_debug_image(self, image: Optional[np.ndarray], label: str):
        if image is None:
            return
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.debug_output_dir / f"{timestamp}_{label}.png"
        try:
            cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f"Saved debug frame to {filename}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"Failed to save debug frame {filename}: {exc}")

    def _cmd_vel_cb(self, msg: Twist):
        """Track last non-zero motion command to detect stalls."""
        if abs(msg.linear.x) > 1e-3 or abs(msg.linear.y) > 1e-3 or abs(msg.angular.z) > 1e-3:
            self.last_motion_time = time.time()
            self.motion_timeout_triggered = False
        if (time.time() - self.last_cmd_vel_log_time) >= self.motion_log_interval:
            self.last_cmd_vel_log_time = time.time()
            self.get_logger().info(
                f"[cmd_vel] lin=({msg.linear.x:.3f},{msg.linear.y:.3f},{msg.linear.z:.3f}), ang=({msg.angular.x:.3f},{msg.angular.y:.3f},{msg.angular.z:.3f})"
            )

    def _odom_cb(self, msg: Odometry):
        """Update when odometry shows the robot actually moving."""
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        lin_speed = math.sqrt(lin.x ** 2 + lin.y ** 2 + lin.z ** 2)
        ang_speed = abs(ang.z)
        self.odom_seen = True
        self.last_odom_msg_time = time.time()
        if lin_speed > self.odom_moving_threshold or ang_speed > 1e-2:
            self.last_odom_motion_time = time.time()
        if (time.time() - self.last_odom_log_time) >= self.motion_log_interval:
            self.last_odom_log_time = time.time()
            self.get_logger().info(f"[odom] lin_speed={lin_speed:.3f}, ang_speed={ang_speed:.3f}")

    def destroy_node(self):
        if self.current_process and self.current_process.poll() is None:
            self.get_logger().info("Terminating active child before shutdown")
            self.current_process.send_signal(signal.SIGINT)
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
        self._join_child_loggers()
        super().destroy_node()


def main():
    runner = ScenarioRunner()
    try:
        rclpy.spin(runner)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            runner.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
