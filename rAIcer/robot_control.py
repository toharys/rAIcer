import os
import serial
import glob
from evdev import InputDevice, categorize, ecodes, list_devices
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque
import queue
import psutil
from enum import Enum

# Configuration
ARDUINO_PORTS = glob.glob('/dev/serial/by-id/*')
KEYBOARD_PATH = "/dev/input/event2"  # Static keyboard path
FRAME_TYPE = 'grayscale'  # 'grayscale', 'depth', or 'color'
STACK_SIZE = 4
MIN_SAMPLE_INTERVAL = 0.05  # 20Hz max sample rate
NYQUIST_MULTIPLIER = 2.5   # Sample 2.5x faster than action changes

class Action(Enum):
    STOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    BACKWARD = 4

class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
   
    def reset(self, frame):
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(frame.copy())
        return self._stack_frames()
   
    def step(self, frame):
        self.frames.append(frame.copy())
        return self._stack_frames()
   
    def _stack_frames(self):
        return np.stack(self.frames, axis=0)

class Robot:
    def __init__(self):
        self.ser = None
        self.keyboard = None
        self.current_action = Action.STOP
        self.previous_action = Action.STOP
        self.frame_stack = FrameStack(STACK_SIZE)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=10)
        self.last_action_change_time = time.time()
        self.last_sample_time = time.time()
        self.action_durations = []
        self.last_key_time = time.time()
        self.debounce_delay = 0.1 
        self.terminate = False
        
        self.keymap = {
            Action.RIGHT: b'w',
            Action.LEFT: b's',
            Action.FORWARD: b'u',
            Action.BACKWARD: b'b',
            Action.STOP: b'x'
        }
        
        self.initialize_connections()
        self.configure_camera()
        
        # Start background threads
        self.feedback_thread = threading.Thread(target=self.read_arduino, daemon=True)
        self.feedback_thread.start()
        self.camera_thread = threading.Thread(target=self._camera_thread_func, daemon=True)
        self.camera_thread.start()
        
        # Set thread affinities if available
        self._set_thread_affinity()
    
    def _set_thread_affinity(self):
        """Set CPU affinity for threads if possible"""
        try:
            p = psutil.Process()
            cores = len(p.cpu_affinity())
            if cores >= 4:
                p.cpu_affinity([0, 1, 2, 3])
                print(f"Set process affinity to cores [0, 1, 2, 3]")
        except Exception as e:
            print(f"Could not set CPU affinity: {e}")

    def initialize_connections(self):
        """Connect to Arduino and keyboard"""
        if not ARDUINO_PORTS:
            raise Exception("No Arduino found")
       
        arduino_port = os.path.realpath(ARDUINO_PORTS[0])
        try:
            self.ser = serial.Serial(arduino_port, 9600, timeout=0.1, write_timeout=0.1)
            print(f"Connected to Arduino on {arduino_port}")
        except Exception as e:
            print(f"Arduino error: {e}")
            exit(1)

        try:
            self.keyboard = InputDevice(KEYBOARD_PATH)
            print(f"Listening to keyboard: {self.keyboard.name}")
        except Exception as e:
            print(f"Keyboard error: {e}")
            exit(1)

    def configure_camera(self):
        """Setup RealSense camera"""
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

    def process_frame(self, frames):
        """Convert frame to desired type"""
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
       
        if FRAME_TYPE == 'grayscale':
            img = np.asanyarray(color_frame.get_data())
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif FRAME_TYPE == 'depth':
            depth = np.asanyarray(depth_frame.get_data())
            return cv2.convertScaleAbs(depth, alpha=0.03)
        elif FRAME_TYPE == 'color':
            return np.asanyarray(color_frame.get_data())
        else:
            raise ValueError("Invalid frame type")

    def frame_processing_binary(self, frame):
        """Process frame to binary black & white"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        if frame_gray.dtype != np.uint8:
            frame_gray = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return clean

    def _camera_thread_func(self):
        """Dedicated thread for camera capture"""
        try:
            while not self.terminate:
                frames = self.pipeline.wait_for_frames()
                processed = self.process_frame(frames)
                
                if processed is not None:
                    current_frame = self.frame_processing_binary(processed) if FRAME_TYPE != 'color' else processed
                    try:
                        self.frame_queue.put(current_frame, timeout=0.1)
                    except queue.Full:
                        continue
        except Exception as e:
            print(f"Camera thread error: {e}")
            self.terminate = True

    def get_frame(self):
        """Get the latest processed frame"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def get_stacked_frames(self):
        """Get stacked frames from the frame stack"""
        frame = self.get_frame()
        if frame is None:
            return None
            
        if not hasattr(self, '_first_frame_received'):
            stacked = self.frame_stack.reset(frame)
            self._first_frame_received = True
        else:
            stacked = self.frame_stack.step(frame)
            
        return stacked

    def read_arduino(self):
        """Continuously read messages from Arduino"""
        while not self.terminate:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline().decode('utf-8').strip()
                    if data:
                        print(f"Arduino: {data}")
            except Exception as e:
                print(f"Error reading from Arduino: {e}")

    def control_with_keyboard(self):
        """Monitor keyboard and update current action"""
        key_map = {
            "KEY_RIGHT": Action.RIGHT,
            "KEY_LEFT": Action.LEFT,
            "KEY_UP": Action.FORWARD,
            "KEY_DOWN": Action.BACKWARD,
            "KEY_SPACE": Action.STOP,
        }
        
        current_keys = set()
        
        try:
            for event in self.keyboard.read_loop():
                if self.terminate:
                    break
                    
                if event.type == ecodes.EV_KEY:
                    key = categorize(event).keycode
                    
                    if event.value == 1:  # Key press
                        if key in key_map:
                            if key not in current_keys:
                                current_keys.add(key)
                                action = key_map[key]
                                self.current_action = action
                                self.ser.write(self.keymap[action])
                        elif key == "KEY_Q":
                            self.terminate = True
                            
                    elif event.value == 0:  # Key release
                        if key in current_keys:
                            current_keys.remove(key)
                            if not current_keys:
                                self.current_action = Action.STOP
                                self.ser.write(self.keymap[Action.STOP])
                            else:
                                last_key = list(current_keys)[-1]
                                action = key_map[last_key]
                                self.ser.write(self.keymap[action])
                                
        except Exception as e:
            print(f"Keyboard thread error: {e}")

    def control_action(self, action):
        """Send command to Arduino based on action"""
        if action in self.keymap:
            self.current_action = action
            self.ser.write(self.keymap[action])
            print(f"Sending: {self.keymap[action].decode()} ({action})")

    def calculate_sample_interval(self):
        """Calculate sampling interval based on action history"""
        if not self.action_durations:
            return MIN_SAMPLE_INTERVAL
        
        recent_durations = self.action_durations[-5:] if len(self.action_durations) >= 5 else self.action_durations
        median_duration = np.median(recent_durations)
        target_interval = median_duration / NYQUIST_MULTIPLIER
        return max(MIN_SAMPLE_INTERVAL, min(target_interval, 1.0))

    def close(self):
        """Clean up resources"""
        self.terminate = True
        if hasattr(self, 'ser') and self.ser:
            self.ser.close()
            print("Serial connection closed")
        
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
            print("Camera stopped")
        
        if self.feedback_thread.is_alive():
            self.feedback_thread.join()
        
        if self.camera_thread.is_alive():
            self.camera_thread.join()