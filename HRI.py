#!/usr/bin/env python3
# encoding: utf-8
import cv2
import time
import rclpy
import queue
import threading
import sys
import numpy as np
import mediapipe as mp
import os
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
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.name = name
        
        # Audio Setup
        self.voice_base = os.environ.get('VOICE_FEEDBACK_PATH') or os.path.join(os.path.dirname(__file__), 'feedback_voice')
        os.environ.setdefault('VOICE_FEEDBACK_PATH', self.voice_base)
        self.voice_enabled = True
        self.voice_cooldown = 1.0
        self.last_voice_played = {}

        # Hand Detector
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        
        # Publishers
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1) 
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)

        # Camera Subscription
        self.camera_topic = '/ascamera/camera_publisher/rgb0/image'
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        from sensor_msgs.msg import Image
        self.create_subscription(Image, self.camera_topic, self.image_callback, 1)

        # State Flags
        self.running = True
        self.fist_detected = False 
        self.wave_detected = False # NEW: Flag for wave
        self.check_attempts = 0 
        
        # Start Threads
        threading.Thread(target=self.image_proc, daemon=True).start()
        threading.Thread(target=self.control_loop, daemon=True).start()
        
        self.get_logger().info('Fist/Wave Node Started. Detects "Fist" (Danger) or "Wave" (Survivor).')

    def _voice_path(self, name: str) -> str:
        base = self.voice_base
        filename = name if os.path.splitext(os.path.basename(name))[1] else name + '.wav'
        if os.path.isabs(filename):
            return filename
        return os.path.join(base, filename)

    def _play_voice(self, name: str, volume: int = 100):
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

    def set_camera_posture(self, mode):
        # 10=Clamp, 5=Wrist, 4=Elbow, 3=Shoulder, 2=Tilt, 1=Pan
        if mode == 'drive':
            positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 500))
            set_servo_position(self.joints_pub, 1.0, positions)
        elif mode == 'look_up':
            positions = ((10, 200), (5, 500), (4, 90), (3, 350), (2, 780), (1, 500))
            set_servo_position(self.joints_pub, 1.0, positions)
        time.sleep(0.5)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        for _ in range(5):
            self.mecanum_pub.publish(twist)
            time.sleep(0.05)

    def move_forward(self):
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
        set_servo_position(self.joints_pub, 1.0, positions)
        twist = Twist()
        twist.angular.z = -1.5 # Negative for opposite direction
        self.mecanum_pub.publish(twist)
        time.sleep(4.2) 
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

    def control_loop(self):
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
            self.get_logger().info("Checking...")
            self.set_camera_posture('look_up')
            
            # 5. Monitor Gestures
            self.get_logger().info("Scanning for Gestures (Fist/Wave)...")
            result = self.check_gestures(2.0)
            
            if result == 'fist':
                self.get_logger().warn("FIST SEEN! (Danger)")
                self._play_voice('Danger') 
                #self.stop_robot()
                self.running = False
                break
            elif result == 'wave':
                self.get_logger().warn("WAVE SEEN! (Survivor)")
                self._play_voice('Survivor') # Make sure survivor.wav exists
                positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 220))
                set_servo_position(self.joints_pub, 1.0, positions)
                time.sleep(0.5)
                positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 780))
                set_servo_position(self.joints_pub, 1.0, positions)
                time.sleep(0.5)
                positions = ((10, 200), (5, 500), (4, 90), (3, 150), (2, 780), (1, 500))
                set_servo_position(self.joints_pub, 1.0, positions)
                self.running = False
                break
            
            else:
                #self.get_logger().info("Nothing detected. Incrementing attempts.")
                #self.check_attempts += 1
                self.rotate_opposite()

                
        #self.stop_robot()
        self.get_logger().info("Exiting Program...")
        rclpy.shutdown()
        sys.exit(0)

    def image_proc(self):
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
            
            self.fist_detected = fist_now
            self.wave_detected = wave_now
            
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