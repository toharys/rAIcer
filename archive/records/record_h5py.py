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
import h5py

# Configuration
KEYBOARD_PATH = "/dev/input/event2"
ARDUINO_PORTS = glob.glob('/dev/serial/by-id/*')
FRAME_TYPE = 'grayscale'  # 'grayscale', 'depth', or 'color'
STACK_SIZE = 4
SAVE_FILENAME = "frame_stacks_with_commands.h5"
TERMINATE = False
SAVE_INTERVAL = 100

def find_keyboard_device():
    """Find the keyboard input device dynamically"""
    keyboard_names = ['keyboard', 'usb keyboard', 'logitech keyboard']
    input_devices = [f"/dev/input/event{i}" for i in range(32)]
   
    for device_path in input_devices:
        try:
            dev = InputDevice(device_path)
            name = dev.name.lower()
            if any(keyword in name for keyword in keyboard_names):
                print(f"Found keyboard: {dev.name} at {device_path}")
                return device_path
        except:
            continue
   
    raise Exception("No keyboard device found.")

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
        """Initialize HDF5 file with proper shapes"""
        self.h5_file = h5py.File(SAVE_FILENAME, 'w')
        
        if FRAME_TYPE == 'color':
            frame_shape = (480, 640, 3)
        else:  # grayscale or depth
            frame_shape = (480, 640)
        
        self.stacks_dataset = self.h5_file.create_dataset(
            'stacks', 
            shape=(0, STACK_SIZE, *frame_shape),
            maxshape=(None, STACK_SIZE, *frame_shape),
            chunks=(1, STACK_SIZE, *frame_shape),
            dtype=np.uint8
        )
        
        self.actions_dataset = self.h5_file.create_dataset(
            'actions',
            shape=(0,),
            maxshape=(None,),
            dtype='S10'
        )
        
        self.next_frames_dataset = self.h5_file.create_dataset(
            'next_frames',
            shape=(0, STACK_SIZE, *frame_shape),
            maxshape=(None, STACK_SIZE, *frame_shape),
            chunks=(1, STACK_SIZE, *frame_shape),
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

    def frame_processing_bw(self, frame):
        """Process frame with contour detection and return processed frame"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        if frame_gray.dtype != np.uint8:
            frame_gray = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        thresh = cv2.adaptiveThreshold(
            frame_gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Contour detection (optional visualization)
        thresh_copy = thresh.copy()
        result = cv2.findContours(thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = result[0] if len(result) == 2 else result[1]
        
        # Return processed frame (with or without contours)
        if FRAME_TYPE == 'color':
            output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
            return output
        else:
            return thresh

    def handle_keyboard(self):
        """Monitor keyboard and update current action"""
        global TERMINATE
        pressed_keys = set()
        
        key_map = {
            "KEY_RIGHT": ('right', b'w'),
            "KEY_LEFT": ('left', b's'),
            "KEY_UP": ('forward', b'u'),
            "KEY_DOWN": ('backward', b'b'),
            "KEY_SPACE": ('stop', b'x'),
        }
        
        try:
            for event in self.keyboard.read_loop():
                if event.type == ecodes.EV_KEY:
                    key = categorize(event).keycode
                    
                    if event.value == 1:  # Key press
                        if key in key_map:
                            pressed_keys.add(key)
                            action, cmd = key_map[key]
                            self.current_action = action
                            self.ser.write(cmd)
                        elif key == "KEY_Q":
                            TERMINATE = True
                    elif event.value == 0:  # Key release
                        if key in pressed_keys:
                            pressed_keys.remove(key)
                        if not pressed_keys:
                            self.current_action = 'stop'
                            self.ser.write(b'x')
                        else:
                            last_key = list(pressed_keys)[-1]
                            action, cmd = key_map[last_key]
                            self.current_action = action
                            self.ser.write(cmd)
        except Exception as e:
            print(f"Keyboard thread error: {e}")

    def save_to_disk(self, stacked_frames, action, next_frames):
        """Save data with proper shape handling"""
        if len(stacked_frames.shape) == 3:
            stacked_frames = np.expand_dims(stacked_frames, axis=0)
        if len(next_frames.shape) == 3:
            next_frames = np.expand_dims(next_frames, axis=0)
        
        new_size = self.sample_count + 1
        self.stacks_dataset.resize(new_size, axis=0)
        self.actions_dataset.resize(new_size, axis=0)
        self.next_frames_dataset.resize(new_size, axis=0)
        
        self.stacks_dataset[self.sample_count] = stacked_frames
        self.actions_dataset[self.sample_count] = np.string_(action)
        self.next_frames_dataset[self.sample_count] = next_frames
        
        self.sample_count += 1
        
        if self.sample_count % SAVE_INTERVAL == 0:
            self.h5_file.flush()

    def capture_data(self, duration=60):
        """Main capture loop with proper frame handling"""
        keyboard_thread = threading.Thread(target=self.handle_keyboard, daemon=True)
        keyboard_thread.start()
        
        # Initialize with first frame
        global TERMINATE
        frames = self.pipeline.wait_for_frames()
        first_frame = None
        while first_frame is None and not TERMINATE:
            frames = self.pipeline.wait_for_frames()
            processed = self.process_frame(frames)
            if processed is not None:
                first_frame = self.frame_processing_bw(processed)
        
        if TERMINATE:
            return
        
        # Initialize frame stack
        stacked_frames = self.frame_stack.reset(first_frame)
        prev_stacked_frames = None
        
        start_time = time.time()
        print(f"Recording for {duration} seconds... (Press 'q' to stop early)")
        
        try:
            while not TERMINATE and (time.time() - start_time < duration):
                frames = self.pipeline.wait_for_frames()
                processed = self.process_frame(frames)
                
                if processed is None:
                    continue
                    
                current_frame = self.frame_processing_bw(processed)
                stacked_frames = self.frame_stack.step(current_frame)
                
                if prev_stacked_frames is not None:
                    self.save_to_disk(prev_stacked_frames, self.current_action, stacked_frames)
                
                prev_stacked_frames = stacked_frames.copy()
                
                cv2.imshow('Current Frame', current_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    TERMINATE = True
                    break
                
                time.sleep(0.05)
                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            if self.ser:
                self.ser.close()
            if self.h5_file:
                self.h5_file.close()
            print(f"Recording complete. Saved {self.sample_count} samples")

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.capture_data(duration=60)