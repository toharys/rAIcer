import serial
import glob
import os
from evdev import InputDevice, categorize, ecodes
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque

# Configuration
KEYBOARD_PATH = "/dev/input/event2"
ARDUINO_PORTS = glob.glob('/dev/serial/by-id/*')
FRAME_TYPE = 'grayscale'  # 'grayscale', 'depth', or 'color'
STACK_SIZE = 4
SAVE_FILENAME = "adar_frame_stacks_with_commands.npz"

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

class DataRecorder:
    def __init__(self):
        self.ser = None
        self.keyboard = None
        self.current_action = 'stop'
        self.recorded_data = []
        self.frame_stack = FrameStack(STACK_SIZE)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
       
        self.initialize_connections()
        self.configure_camera()
       
    def initialize_connections(self):
        """Connect to Arduino and keyboard"""
        if not ARDUINO_PORTS:
            raise Exception("No Arduino found")
       
        arduino_port = os.path.realpath(ARDUINO_PORTS[0])
        try:
            self.ser = serial.Serial(arduino_port, 9600, timeout=1)
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

    def handle_keyboard(self):
        """Monitor keyboard and update current action"""
        try:
            for event in self.keyboard.read_loop():
                if event.type == ecodes.EV_KEY and event.value == 1:
                    key = categorize(event).keycode
                   
                    if key == "KEY_RIGHT":
                        self.current_action = 'right'
                        self.ser.write(b'w')
                    elif key == "KEY_LEFT":
                        self.current_action = 'left'
                        self.ser.write(b's')
                    elif key == "KEY_UP":
                        self.current_action = 'forward'
                        self.ser.write(b'u')
                    elif key == "KEY_DOWN":
                        self.current_action = 'backward'
                        self.ser.write(b'b')
                    elif key == "KEY_SPACE":
                        self.current_action = 'stop'
                        self.ser.write(b'x')
        except Exception as e:
            print(f"Keyboard thread error: {e}")

    def capture_data(self, duration=60):
        """Main capture loop"""
        keyboard_thread = threading.Thread(target=self.handle_keyboard, daemon=True)
        keyboard_thread.start()
       
        # Initialize with first frame
        frames = self.pipeline.wait_for_frames()
        first_frame = self.process_frame(frames)
        if first_frame is None:
            raise ValueError("Camera initialization failed")
        self.frame_stack.reset(first_frame)
       
        start_time = time.time()
        print(f"Recording for {duration} seconds... (Press 'q' to stop early)")
       
        try:
            t = 0
            terminate = False
            while (time.time() - start_time) < duration:
                frames = self.pipeline.wait_for_frames()
                current_frame = self.process_frame(frames)
               
                if current_frame is None:
                    continue
               
                # Get frame stack and record with current action
                stacked_frames = self.frame_stack.step(current_frame)
                self.recorded_data.append([stacked_frames.copy(), self.current_action])
               
                # Display
                cv2.imshow('Current Frame', current_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
               
                # Reset to stop if no recent keypress
                if time.time() - start_time > 0.5:  # 0.5s timeout
                    self.current_action = 'stop'
               
                time.sleep(0.05)  # Control frame rate
           
            self.save_data()
           
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.ser.close()
            print(f"Recording complete. Saved {len(self.recorded_data)} samples")

    def save_data(self):
        """Save recorded data to file"""
        np.savez_compressed(
            SAVE_FILENAME,
            stacks=[d[0] for d in self.recorded_data],
            actions=[d[1] for d in self.recorded_data]
        )
        print(f"Data saved to {SAVE_FILENAME}")

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.capture_data(duration=20)  # Record for 60 seconds
