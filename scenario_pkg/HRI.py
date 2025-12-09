#!/usr/bin/env python3
# encoding: utf-8
"""Gesture-based safety/human-interaction helper.

This node watches the camera feed for a closed fist (danger) or a wave/open
palm (survivor/all-clear). When a gesture is confirmed it can stop the robot,
play a short voice clip, and move the camera into a posture that fits the
situation. Comments throughout the file use plain language to describe the
flow.
"""

import cv2
import time
import rclpy
import queue
import threading
import sys
import numpy as np
import mediapipe as mp
import os
try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position

# --- IMPORTS FOR AUDIO ---
from speech import speech
# -------------------------

# MediaPipe constants
mp_hands = mp.solutions.hands
WRIST = mp_hands.HandLandmark.WRIST
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_FINGER_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
RING_FINGER_TIP = mp_hands.HandLandmark.RING_FINGER_TIP
PINKY_TIP = mp_hands.HandLandmark.PINKY_TIP

INDEX_FINGER_MCP = mp_hands.HandLandmark.INDEX_FINGER_MCP
MIDDLE_FINGER_MCP = mp_hands.HandLandmark.MIDDLE_FINGER_MCP
RING_FINGER_MCP = mp_hands.HandLandmark.RING_FINGER_MCP
PINKY_MCP = mp_hands.HandLandmark.PINKY_MCP

class FistStopNode(Node):
    """ROS 2 node that turns hand gestures into simple robot actions."""

    def __init__(self, name):
        # This node usually runs on its own, so we initialize rclpy here. If it
        # is ever embedded in a larger app, have that app call ``rclpy.init``
        # before constructing this node to avoid double-initialization errors.
        if not rclpy.ok():
            rclpy.init()
        super().__init__(name)
        self.name = name

        # Audio Setup
        self.voice_base = self._resolve_voice_base()
        self.voice_enabled = bool(self.voice_base)
        self.voice_cooldown = 1.0
        self.last_voice_played = {}
        if self.voice_base:
            os.environ.setdefault('VOICE_FEEDBACK_PATH', self.voice_base)
            self.get_logger().info(f"Voice feedback path: {self.voice_base}")
            self._log_voice_files()
        else:
            self.get_logger().warn("Voice feedback disabled: no feedback_voice directory found. "
                                   "Set VOICE_FEEDBACK_PATH to a folder containing .wav files (e.g., Danger.wav, Survivor.wav).")

        # Hand Detector
        # MediaPipe is used to spot hands and the individual finger joints. We
        # keep the thresholds moderate so noisy video still registers gestures
        # without requiring perfect lighting.
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        
        # Publishers
        # These outputs let the node stop the drive base and reposition the arm
        # servos when a gesture is confirmed.
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self._servo_log_cooldown = 0.50  # seconds between logging servo commands
        self._last_servo_log = 0.0

        # Camera Subscription
        # Images are queued so the processing thread always sees the freshest
        # frame without blocking the ROS callback.
        self.camera_topic = '/ascamera/camera_publisher/rgb0/image'
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        from sensor_msgs.msg import Image
        self.create_subscription(Image, self.camera_topic, self.image_callback, 1)

        # State Flags
        # These booleans track what the latest video frame showed and whether
        # the node should keep running.
        self.running = True
        self.fist_detected = False
        self.wave_detected = False # NEW: Flag for wave
        self.check_attempts = 0
        self.fist_hold_start = None
        self.wave_hold_start = None

        # Start Threads
        # A lightweight pipeline: one thread converts images to gestures, the
        # other decides what robot action to take.
        threading.Thread(target=self.image_proc, daemon=True).start()
        threading.Thread(target=self.control_loop, daemon=True).start()
        
        self.get_logger().info('Fist/Wave Node Started. Detects "Fist" (Danger) or "Wave" (Survivor).')

    def _resolve_voice_base(self):
        """
        Resolve where to load audio files from.
        Priority: VOICE_FEEDBACK_PATH env -> package share feedback_voice -> local feedback_voice next to this file.
        """
        candidates = []
        # 1) Prefer folder next to this file (works in source and install space)
        candidates.append(os.path.join(os.path.dirname(__file__), 'feedback_voice'))

        # 2) Explicit env override
        env_path = os.environ.get('VOICE_FEEDBACK_PATH')
        if env_path:
            candidates.append(env_path)

        # 3) Package share (colcon install space)
        if get_package_share_directory:
            try:
                pkg_share = get_package_share_directory('HRI_pkg')
                candidates.append(os.path.join(pkg_share, 'feedback_voice'))
            except Exception:
                pass

        for path in candidates:
            if path and os.path.isdir(path):
                return path
        return None

    def _voice_path(self, name: str) -> str:
        base = self.voice_base
        filename = name if os.path.splitext(os.path.basename(name))[1] else name + '.wav'
        if os.path.isabs(filename):
            return filename
        return os.path.join(base, filename)

    def _log_voice_files(self):
        """Log which wav files are available to help debug missing audio."""
        if not self.voice_base:
            return
        try:
            files = [f for f in os.listdir(self.voice_base) if f.lower().endswith('.wav')]
        except FileNotFoundError:
            self.get_logger().warn(f"Voice path does not exist: {self.voice_base}")
            return

        if not files:
            self.get_logger().warn(f"No .wav files found in {self.voice_base}. "
                                   "Expected files like Danger.wav and Survivor.wav.")
        else:
            self.get_logger().info(f"Available voice files: {', '.join(files)}")

    def _play_voice(self, name: str, volume: int = 100):
        """Play a short wav file if voice is enabled and not on cooldown."""
        if not self.voice_enabled:
            return
        path = self._voice_path(name)
        now = time.time()
        last_played = self.last_voice_played.get(path)
        if last_played is not None and (now - last_played) < self.voice_cooldown:
            return
        try:
            if os.path.exists(path):
                speech.set_volume(volume)
                speech.play_audio(path)
                self.last_voice_played[path] = now
            else:
                self.get_logger().warn(f"Audio file not found: {path}")
        except Exception as e:
            self.get_logger().error(f"Voice playback failed for {name}: {e}")

    def image_callback(self, ros_image):
        """Convert incoming ROS images to RGB frames and enqueue them."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
            rgb_image = np.array(cv_image, dtype=np.uint8)
            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(rgb_image)
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def is_fist(self, landmarks, shape):
        """Detects if 3 or more fingers are folded (Fist)"""
        h, w, _ = shape
        def get_coord(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        wrist = get_coord(WRIST)
        fingers_indices = [
            (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
            (RING_FINGER_TIP, RING_FINGER_MCP),
            (PINKY_TIP, PINKY_MCP)
        ]
        
        folded_count = 0
        for tip_idx, mcp_idx in fingers_indices:
            tip = get_coord(tip_idx)
            mcp = get_coord(mcp_idx)
            # Tip closer to wrist than knuckle = Folded
            if np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist) * 1.2: 
                folded_count += 1
        return folded_count >= 3

    def is_wave(self, landmarks, shape):
        """Detects if 4 or more fingers are extended (Open Palm / Wave)"""
        h, w, _ = shape
        def get_coord(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        wrist = get_coord(WRIST)
        fingers_indices = [
            (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
            (RING_FINGER_TIP, RING_FINGER_MCP),
            (PINKY_TIP, PINKY_MCP),
            (THUMB_TIP, mp_hands.HandLandmark.THUMB_CMC) # Added thumb for wave
        ]
        
        extended_count = 0
        for tip_idx, mcp_idx in fingers_indices:
            tip = get_coord(tip_idx)
            mcp = get_coord(mcp_idx)
            # Tip further from wrist than knuckle = Extended
            if np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist): 
                extended_count += 1
        return extended_count >= 4

    def _set_servo_position(self, positions, duration=1.0, label=None):
        """Send a servo command and log it no more than every 250 ms."""
        now = time.time()
        if (now - self._last_servo_log) >= self._servo_log_cooldown:
            tag = f" {label}" if label else ""
            self.get_logger().info(f"[servo]{tag}: {positions}")
            self._last_servo_log = now
        set_servo_position(self.joints_pub, duration, positions)

    def set_camera_posture(self, mode):
        """Move the camera servos into sensible poses for each action."""
        # 10=Clamp, 5=Wrist, 4=Elbow, 3=Shoulder, 2=Tilt, 1=Pan
        if mode == 'drive':
            positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 500))
            self._set_servo_position(positions, 1.0, "drive_pose")
        elif mode == 'look_up':
            positions = ((10, 200), (5, 500), (4, 90), (3, 350), (2, 780), (1, 500))
            self._set_servo_position(positions, 1.0, "look_up_pose")
        elif mode == 'look_down':
            positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 550), (1, 500))
            self._set_servo_position(positions, 1.0, "look_down_pose")
        time.sleep(0.5)

    def stop_robot(self):
        """Send several zero-velocity commands to ensure the base stops."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        for _ in range(5):
            self.mecanum_pub.publish(twist)
            time.sleep(0.05)

    def move_forward(self):
        """Creep forward slowly; used when approaching gestures."""
        twist = Twist()
        twist.linear.x = 0.15
        self.mecanum_pub.publish(twist)

    def rotate_once(self):
        """Timeout rotation: Left (Positive Z)"""
        self.get_logger().warn("Attempts limit reached. Rotating once (Left)...")
        twist = Twist()
        twist.angular.z = 1.5
        self.mecanum_pub.publish(twist)
        time.sleep(4.2) 
        self.stop_robot()

    def rotate_opposite(self):
        """Survivor rotation: Right (Negative Z), Opposite to rotate_once"""
        self.get_logger().warn("Survivor found. Rotating Opposite (Right)...")
        positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 500))
        self._set_servo_position(positions, 1.0, "rotate_opposite")
        twist = Twist()
        twist.angular.z = -1.5 # Negative for opposite direction
        self.mecanum_pub.publish(twist)
        time.sleep(4.2) 
        self.stop_robot()

    def rotate_quarter_turn(self):
        """Rotate roughly 90 degrees to continue scanning in a new sector."""
        twist = Twist()
        twist.angular.z = 1.5
        self.get_logger().info("Rotating 90 degrees to continue scanning...")
        self.mecanum_pub.publish(twist)
        time.sleep(1.1)  # ~pi/2 at 1.5 rad/s
        self.stop_robot()

    def check_gestures(self, duration):
        """
        Polls for gestures. Returns 'fist', 'wave', or None.
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.fist_detected:
                return 'fist'
            if self.wave_detected:
                return 'wave'
            time.sleep(0.05)
        return None

    def scan_with_arm(self):
        """
        Sweep the camera with pan/tilt instead of rotating the base.
        Returns 'fist', 'wave', or None based on what is seen during the sweep.
        """
        base_pose = ((10, 200), (5, 500), (4, 90), (3, 350))
        pan_tilt_sequence = [
            (500, 780),  # center up
            (220, 780),  # left up
            (780, 780),  # right up
            (500, 700),  # center mid
            (220, 700),  # left mid
            (780, 700),  # right mid
        ]

        for pan, tilt in pan_tilt_sequence:
            positions = base_pose + ((2, tilt), (1, pan))
            self.get_logger().info(f"Scanning pan={pan}, tilt={tilt}")
            self._set_servo_position(positions, 1.0, "scan")
            time.sleep(0.3)  # allow movement to settle
            result = self.check_gestures(1.5)
            if result:
                return result
        return None

    def control_loop(self):
        """High-level behavior that reacts to the latest gesture flags."""
        time.sleep(2)
        
        while self.running:
            #if self.check_attempts >= 3:
            #    self.rotate_once()
            #    self.stop_robot()
            #    self.running = False
            #    break

            # 1. Prepare to Drive
            #self.set_camera_posture('drive')
            
            # 2. Move Forward
            #self.get_logger().info(f"Moving Forward ({self.check_attempts + 1}/3)...")
            #self.move_forward()
            #time.sleep(3.0) 
            
            # 3. Stop
            #self.stop_robot()
            
            # 4. Look Up / Check
            self.get_logger().info("Scanning for Gestures (Fist/Wave) with arm pan/tilt...")
            result = self.scan_with_arm()
            
            if result == 'fist':
                self.get_logger().warn("FIST SEEN! (Danger)")
                self._play_voice('Danger') 
                self.set_camera_posture('look_down')
                self.stop_robot()
                self.running = False
                break
            elif result == 'wave':
                self.get_logger().warn("WAVE SEEN! (Survivor)")
                self._play_voice('Survivor') # Make sure survivor.wav exists
                # Keep tilt level (no downward tilt) and speed up sweep
                positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 220))  # left
                self._set_servo_position(positions, 1.0, "wave_left")
                time.sleep(0.25)
                positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 780))  # right
                self._set_servo_position(positions, 1.0, "wave_right")
                time.sleep(0.25)
                positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 220))  # back left
                self._set_servo_position(positions, 1.0, "wave_back_left")
                time.sleep(0.25)
                positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 500))  # center
                self._set_servo_position(positions, 1.0, "wave_center")
                # Return to scanning posture and pause before resuming search
                self.set_camera_posture('look_up')
                time.sleep(3.0)
            
            else:
                self.get_logger().info("No gestures detected. Rotating and retrying.")
                self.rotate_quarter_turn()

                
        #self.stop_robot()
        self.get_logger().info("Exiting Program...")
        rclpy.shutdown()
        sys.exit(0)

    def image_proc(self):
        """Continuously convert images into gesture flags for the control loop."""
        while self.running:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            image_flip = cv2.flip(image, 1)
            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)
            results = self.hand_detector.process(image_flip)
            
            # Reset local flags
            fist_now = False
            wave_now = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(
                        bgr_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if self.is_fist(hand_landmarks.landmark, image_flip.shape):
                        fist_now = True
                        cv2.putText(bgr_image, "FIST (DANGER)", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif self.is_wave(hand_landmarks.landmark, image_flip.shape):
                        wave_now = True
                        cv2.putText(bgr_image, "WAVE (SURVIVOR)", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            now = time.time()
            # Require ~200ms of continuous presence to reduce sensitivity
            if fist_now:
                if self.fist_hold_start is None:
                    self.fist_hold_start = now
                self.fist_detected = (now - self.fist_hold_start) >= 0.2
            else:
                self.fist_hold_start = None
                self.fist_detected = False

            if wave_now:
                if self.wave_hold_start is None:
                    self.wave_hold_start = now
                self.wave_detected = (now - self.wave_hold_start) >= 0.2
            else:
                self.wave_hold_start = None
                self.wave_detected = False
            
            cv2.imshow(self.name, bgr_image)
            key = cv2.waitKey(1)
            if key == 27:
                self.running = False

def main():
    node = FistStopNode('fist_back_node')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        node.destroy_node()

if __name__ == "__main__":
    main()
