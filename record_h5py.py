import os
os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '')}:~/move_robot/rAIcer/librealsense/build/Release"

import serial
import glob
from evdev import InputDevice, categorize, ecodes
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque
import h5py  # For efficient disk-based storage

# Configuration
KEYBOARD_PATH = "/dev/input/event3"
ARDUINO_PORTS = glob.glob('/dev/serial/by-id/*')
FRAME_TYPE = 'grayscale'  # 'grayscale', 'depth', or 'color'
STACK_SIZE = 4
SAVE_FILENAME = "time_0.05_frame_stacks_with_commands.h5"  # Changed to HDF5 format
TERMINATE = False
SAVE_INTERVAL = 100  # Save every 100 samples to disk

def find_keyboard_device():
    """Find the keyboard input device dynamically"""
    # Common keyboard identifiers - you may need to add more
    keyboard_names = ['keyboard', 'usb keyboard', 'logitech keyboard']
   
    # Check all input devices
    input_devices = [f"/dev/input/event{i}" for i in range(32)]  # Check up to event32
   
    for device_path in input_devices:
        try:
            dev = InputDevice(device_path)
            name = dev.name.lower()
            if any(keyword in name for keyword in keyboard_names):
                print(f"Found keyboard: {dev.name} at {device_path}")
                return device_path
        except:
            continue
   
    raise Exception("No keyboard device found. Check if keyboard is connected.")

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
        self.frame_stack = FrameStack(STACK_SIZE)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.h5_file = None
        self.stacks_dataset = None
        self.actions_dataset = None
        self.next_frames_dataset = None
        self.sample_count = 0
       
        self.initialize_connections()
        self.configure_camera()
        self.initialize_storage()
       
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
            #keyboard_path = find_keyboard_device()
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

    def initialize_storage(self):
        """Initialize HDF5 file for storage"""
        self.h5_file = h5py.File(SAVE_FILENAME, 'w')
        
        # Get frame shape (adjust based on your frame type)
        if FRAME_TYPE == 'color':
            frame_shape = (480, 640, 3)
        else:  # grayscale or depth
            frame_shape = (480, 640)
        
        # Create resizable datasets
        self.stacks_dataset = self.h5_file.create_dataset(
            'stacks', 
            shape=(0, STACK_SIZE, *frame_shape),
            maxshape=(None, STACK_SIZE, *frame_shape),
            chunks=(SAVE_INTERVAL, STACK_SIZE, *frame_shape),
            dtype=np.uint8
        )
        
        # For older h5py versions, we'll store actions as fixed-length ASCII strings
        self.actions_dataset = self.h5_file.create_dataset(
            'actions',
            shape=(0,),
            maxshape=(None,),
            dtype='S10'  # Fixed-length string type (10 characters max)
        )
        
        self.next_frames_dataset = self.h5_file.create_dataset(
            'next_frames',
            shape=(0, STACK_SIZE, *frame_shape),
            maxshape=(None, STACK_SIZE, *frame_shape),
            chunks=(SAVE_INTERVAL, STACK_SIZE, *frame_shape),
            dtype=np.uint8
        )

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
        global TERMINATE
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
                        time.sleep(0.1)
                        self.ser.write(b'x')
                    elif key == "KEY_DOWN":
                        self.current_action = 'backward'
                        self.ser.write(b'b')
                        time.sleep(0.1)
                        self.ser.write(b'x')
                    elif key == "KEY_SPACE":
                        self.current_action = 'stop'
                        self.ser.write(b'x')
                    elif key == "KEY_Q":
                        TERMINATE = True
        except Exception as e:
            print(f"Keyboard thread error: {e}")

    def save_to_disk(self, stacked_frames, action, next_frames):
        """Save data to HDF5 file incrementally"""
        # Resize datasets
        new_size = self.sample_count + 1
        self.stacks_dataset.resize(new_size, axis=0)
        self.actions_dataset.resize(new_size, axis=0)
        self.next_frames_dataset.resize(new_size, axis=0)
        
        # Add data
        self.stacks_dataset[self.sample_count] = stacked_frames
        self.actions_dataset[self.sample_count] = np.string_(action)  # Convert to bytes
        self.next_frames_dataset[self.sample_count] = next_frames
        
        self.sample_count += 1
        
        # Flush periodically
        if self.sample_count % SAVE_INTERVAL == 0:
            self.h5_file.flush()

    def capture_data(self, duration=60):
        """Main capture loop"""
        keyboard_thread = threading.Thread(target=self.handle_keyboard, daemon=True)
        keyboard_thread.start()
       
        # Initialize with first frame
        global TERMINATE
        frames = self.pipeline.wait_for_frames()
        first_frame = self.process_frame(frames)
        if first_frame is None:
            raise ValueError("Camera initialization failed")
        self.frame_stack.reset(first_frame)
       
        start_time = time.time()
        print(f"Recording for {duration} seconds... (Press 'q' to stop early)")
       
        try:
            prev_stacked_frames = None
           
            while not TERMINATE:
                frames = self.pipeline.wait_for_frames()
                current_frame = self.process_frame(frames)
               
                if current_frame is None:
                    continue
               
                # Get frame stack
                stacked_frames = self.frame_stack.step(current_frame)
               
                # Save previous frame with its action and current frame as next state
                if prev_stacked_frames is not None:
                    self.save_to_disk(prev_stacked_frames, self.current_action, stacked_frames)
               
                prev_stacked_frames = stacked_frames.copy()
               
                # Display
                cv2.imshow('Current Frame', current_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    TERMINATE = True
                    break
               
                # Reset to stop if no recent keypress
                if time.time() - start_time > 0.5:  # 0.5s timeout
                    self.current_action = 'stop'
               
                time.sleep(0.1)  # Control frame rate
            self.h5_file.flush()
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.ser.close()
            self.h5_file.close()
            print(f"Recording complete. Saved {self.sample_count} samples")

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.capture_data(duration=60)  # Record for 60 seconds
