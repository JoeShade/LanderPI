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
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from std_srvs.srv import SetBool
from interfaces.srv import SetString

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# LAB bounds for the default black line (hard-coded so no color picking is
# required).
BLACK_LAB_RANGE = {"min": [0, 0, 0], "max": [40, 120, 120]}
GREEN_LAB_RANGE = {"min": [0, 80, 0], "max": [255, 120, 120]}  # broad green

# How many frames without a line before we transition away from line following.
LINE_LOST_FRAMES = 60

# How large the green beacon should appear (fraction of the image area) before
# handing control to HRI.
BEACON_AREA_THRESHOLD = 0.06

# Debugging defaults.
DEBUG_LOG_EVERY_N_FRAMES = 15
DEBUG_OUTPUT_DIR = Path("/tmp/scenario_debug")

# Region-of-interest stack matching the camera choices in line_following.py to
# reuse its tuning. These percentages are (y_min, y_max, x_min, x_max, weight).
ROI_TABLE = {
    "ascamera": ((0.9, 0.95, 0, 1, 0.7), (0.8, 0.85, 0, 1, 0.2), (0.7, 0.75, 0, 1, 0.1)),
    "aurora": ((0.81, 0.83, 0, 1, 0.7), (0.69, 0.71, 0, 1, 0.2), (0.57, 0.59, 0, 1, 0.1)),
    "usb_cam": ((0.79, 0.81, 0, 1, 0.7), (0.67, 0.69, 0, 1, 0.2), (0.55, 0.57, 0, 1, 0.1)),
}


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

    @staticmethod
    def _largest_contour(contours, threshold=30):
        contour_area = zip(contours, tuple(map(lambda c: abs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if contour_area:
            return max(contour_area, key=lambda c_a: c_a[1])
        return None

    def detect_angle(self, image: np.ndarray, lowerb, upperb) -> Optional[float]:
        """Return the steering angle if a line is found, None otherwise."""

        h, w = image.shape[:2]
        centroid_sum = 0.0
        for roi in self.rois:
            blob = image[int(roi[0] * h): int(roi[1] * h), int(roi[2] * w): int(roi[3] * w)]
            mask = cv2.inRange(cv2.cvtColor(blob, cv2.COLOR_RGB2LAB), lowerb, upperb)
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            max_contour_area = self._largest_contour(contours)
            if max_contour_area is not None:
                rect = cv2.minAreaRect(max_contour_area[0])
                box = np.intp(cv2.boxPoints(rect))
                center_x = (box[0, 0] + box[2, 0]) / 2
                center_y = (box[0, 1] + box[2, 1]) / 2
                cv2.circle(blob, (int(center_x), int(center_y)), 3, (255, 0, 0), -1)
                centroid_sum += center_x * roi[-1]

        if centroid_sum == 0:
            return None
        center_pos = centroid_sum / max(self.weight_sum, 1e-6)
        deflection_angle = -math.atan((center_pos - (w / 2.0)) / (h / 2.0))
        return deflection_angle


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
        self.camera_topic = "/ascamera/camera_publisher/rgb0/image"
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self._image_cb, qos)

        # Resolve the installed share dir so we can launch the original scripts even from install space.
        self.share_dir = Path(get_package_share_directory("scenario_pkg"))

        # Stage tracking.
        self.stage = Stage.LINE
        self.current_process: Optional[subprocess.Popen] = None
        self.line_lost_frames = 0
        self.frame_counter = 0
        self.last_image: Optional[np.ndarray] = None
        self.child_log_threads: Tuple[threading.Thread, ...] = tuple()
        camera_type = os.environ.get("DEPTH_CAMERA_TYPE", "aurora")
        self.line_watcher = LineWatcher(ROI_TABLE.get(camera_type, ROI_TABLE["aurora"]))
        self.green_watcher = GreenWatcher()

        # Precompute LAB bounds used for transition detection.
        self.black_lower = tuple(BLACK_LAB_RANGE["min"])
        self.black_upper = tuple(BLACK_LAB_RANGE["max"])
        self.green_lower = tuple(GREEN_LAB_RANGE["min"])
        self.green_upper = tuple(GREEN_LAB_RANGE["max"])

        # Allow external introspection when diagnosing issues.
        self.create_service(Trigger, "get_status", self._handle_get_status)

        # Start the first stage immediately.
        self._launch_stage(Stage.LINE)

        # Periodic heartbeat to log what is active.
        self.create_timer(5.0, self._log_status)

    # ------------------------------------------------------------------
    # Transition handling
    # ------------------------------------------------------------------
    def _launch_stage(self, stage: Stage):
        """Start the requested stage's original script as a subprocess."""

        self._stop_child_process()

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
        self.current_process = subprocess.Popen(
            [sys.executable, "-u", script], env=env, stdout=stdout_pipe, stderr=stderr_pipe, text=True, bufsize=1
        )
        self.stage = stage
        self.line_lost_frames = 0
        self.frame_counter = 0
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
            except subprocess.TimeoutExpired:
                self.get_logger().warning("Force-killing previous stage")
                self.current_process.kill()
        self._join_child_loggers()

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
        return getattr(future.result(), "success", True)

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
        self._call_service(SetString, f"{base_name}/set_color", SetString.Request(data="black"))
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

    # ------------------------------------------------------------------
    # Image callbacks used purely for transition decisions
    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image):
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

    def _handle_line_stage(self, image: np.ndarray):
        angle = self.line_watcher.detect_angle(image, self.black_lower, self.black_upper)
        if angle is None:
            self.line_lost_frames += 1
        else:
            self.line_lost_frames = 0

        if self.debug_mode and self.frame_counter % self.debug_log_every_n == 0:
            self.get_logger().info(
                f"[LINE debug] frame={self.frame_counter}, angle={angle if angle is not None else 'none'}, "
                f"lost_frames={self.line_lost_frames}"
            )

        if self.line_lost_frames >= LINE_LOST_FRAMES:
            self.get_logger().info("Line lost for %d frames, switching to GREEN stage" % self.line_lost_frames)
            if self.save_debug_images:
                self._save_debug_image(image, "line_lost")
            self._launch_stage(Stage.GREEN)

    def _handle_green_stage(self, image: np.ndarray):
        area_ratio = self.green_watcher.area_ratio(image, self.green_lower, self.green_upper)
        if self.debug_mode and self.frame_counter % self.debug_log_every_n == 0:
            self.get_logger().info(f"[GREEN debug] frame={self.frame_counter}, beacon_area={area_ratio:.4f}")
        if area_ratio >= BEACON_AREA_THRESHOLD:
            self.get_logger().info(
                f"Beacon reached (area ratio {area_ratio:.3f} >= {BEACON_AREA_THRESHOLD}), switching to HRI"
            )
            if self.save_debug_images:
                self._save_debug_image(image, "beacon_reached")
            self._launch_stage(Stage.HRI)

    # ------------------------------------------------------------------
    # Logging and shutdown
    # ------------------------------------------------------------------
    def _log_status(self):
        running = self.current_process and self.current_process.poll() is None
        self.get_logger().info(f"Stage={self.stage.name}, child_running={running}, lost_frames={self.line_lost_frames}")

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
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
