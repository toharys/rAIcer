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
import h5py
import queue
import psutil

# Configuration
ARDUINO_PORTS = glob.glob('/dev/serial/by-id/*')
KEYBOARD_PATH = "/dev/input/event2"  # Static keyboard path
FRAME_TYPE = 'grayscale'  # 'grayscale', 'depth', or 'color'
STACK_SIZE = 4
SAVE_FILENAME =  "/media/raicer/7351-8BE8/only_line_0.h5"
SAVE_INTERVAL = 300
MIN_SAMPLE_INTERVAL = 0.05  # 20Hz max sample rate
NYQUIST_MULTIPLIER = 2.5   # Sample 2.5x faster than action changes

def set_thread_affinity(thread, cores):
    """Set CPU affinity for the entire process (works reliably in Python 3.6)"""
    try:
        p = psutil.Process()
        p.cpu_affinity(cores)
        print(f"Set process affinity to cores {cores}")
    except Exception as e:
        print(f"Could not set CPU affinity: {e}")

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
        self.previous_action = 'stop'
        self.frame_stack = FrameStack(STACK_SIZE)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.h5_file = None
        self.stacks_dataset = None
        self.actions_dataset = None
        self.next_frames_dataset = None
        self.sample_count = 0
        self.serial_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=10)  # Buffer for frames
        self.save_queue = queue.Queue(maxsize=5)    # Buffer for saving
        self.last_action_change_time = time.time()
        self.last_sample_time = time.time()
        self.action_durations = []
        self.last_key_time = time.time()
        self.debounce_delay = 0.1 
        self.terminate = False
        self.initialize_connections()
        self.configure_camera()
        self.initialize_storage()
       
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
            print("Available keys:", self.keyboard.capabilities().get(1, []))
        except Exception as e:
            print(f"Keyboard error: {e}")
            exit(1)

    def configure_camera(self):
        """Setup RealSense camera"""
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

    def initialize_storage(self):
        """Initialize HDF5 file"""
        self.h5_file = h5py.File(SAVE_FILENAME, 'w')
        frame_shape = (480, 640, 3) if FRAME_TYPE == 'color' else (480, 640)
        
        self.stacks_dataset = self.h5_file.create_dataset(
            'stacks', 
            shape=(0, STACK_SIZE, *frame_shape),
            maxshape=(None, STACK_SIZE, *frame_shape),
            chunks=(1, STACK_SIZE, *frame_shape),
            dtype=np.uint8,
            compression='lzf'
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
            dtype=np.uint8,
            compression='lzf'
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

    def frame_processing_binary(self, frame):
        """Process frame to binary black & white"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        if frame_gray.dtype != np.uint8:
            frame_gray = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        """        
	thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        """
        _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #edges = cv2.Canny(clean, 50, 150)
        return clean 

    def handle_keyboard(self):
        """Monitor keyboard and update current action"""
        key_map = {
            "KEY_RIGHT": ('right', b'w'),
            "KEY_LEFT": ('left', b's'),
            "KEY_UP": ('forward', b'u'),
            "KEY_DOWN": ('backward', b'b'),
            "KEY_SPACE": ('stop', b'x'),
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
                                action, cmd = key_map[key]
                                self.current_action = action
                                self.ser.write(cmd)
                        elif key == "KEY_Q":
                            self.terminate = True
                            
                    elif event.value == 0:  # Key release
                        if key in current_keys:
                            current_keys.remove(key)
                            if not current_keys:
                                self.current_action = 'stop'
                                self.ser.write(b'x')
                            else:
                                last_key = list(current_keys)[-1]
                                action, cmd = key_map[last_key]
                                self.ser.write(cmd)
                                
        except Exception as e:
            print(f"Keyboard thread error: {e}")  


    def camera_thread(self):
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

    def processing_thread(self):
        """Thread for processing frames and preparing data for saving"""
        try:
            # Initialize frame stack
            first_frame = None
            while first_frame is None and not self.terminate:
                try:
                    first_frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            
            if self.terminate:
                return
            
            stacked_frames = self.frame_stack.reset(first_frame)
            prev_stacked_frames = None
            prev_action = 'stop'
            
            while not self.terminate:
                current_time = time.time()
                sample_interval = self.calculate_sample_interval()
                
                if current_time - self.last_sample_time >= sample_interval:
                    try:
                        current_frame = self.frame_queue.get(timeout=0.1)
                        stacked_frames = self.frame_stack.step(current_frame)
                        
                        if prev_stacked_frames is not None:
                            self.save_queue.put((prev_stacked_frames.copy(), prev_action, stacked_frames.copy()))
                        
                        prev_stacked_frames = stacked_frames.copy()
                        prev_action = self.current_action
                        self.last_sample_time = current_time
                        
                        cv2.imshow('Current Frame', current_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.terminate = True
                            break
                    except queue.Empty:
                        continue
                
                sleep_time = max(0, (self.last_sample_time + sample_interval) - time.time())
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"Processing thread error: {e}")
            self.terminate = True
        finally:
            cv2.destroyAllWindows()

    def save_thread(self):
        """Dedicated thread for saving data to disk"""
        try:
            while not self.terminate:
                try:
                    stacked_frames, action, next_frames = self.save_queue.get(timeout=0.1)
                    self.save_to_disk(stacked_frames, action, next_frames)
                except queue.Empty:
                    time.sleep(0.005)  # Small sleep to prevent CPU spin - to not block the main thtrad
        except Exception as e:
            print(f"Save thread error: {e}")
            self.terminate = True

    def calculate_sample_interval(self):
        """Calculate sampling interval based on action history"""
        if not self.action_durations:
            return MIN_SAMPLE_INTERVAL
        
        recent_durations = self.action_durations[-5:] if len(self.action_durations) >= 5 else self.action_durations
        median_duration = np.median(recent_durations)
        target_interval = median_duration / NYQUIST_MULTIPLIER
        return max(MIN_SAMPLE_INTERVAL, min(target_interval, 1.0))

    def save_to_disk(self, stacked_frames, action, next_frames):
        """Save data to HDF5 file"""
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
        """Main capture loop with debugging"""
        # Create and start all threads
        keyboard_thread = threading.Thread(target=self.handle_keyboard, daemon=True, name="keyboard_thread")
        camera_thread = threading.Thread(target=self.camera_thread, daemon=True, name="camera_thread")
        processing_thread = threading.Thread(target=self.processing_thread, daemon=True, name="processing_thread")
        save_thread = threading.Thread(target=self.save_thread, daemon=True, name="save_thread")
        
        # Start all threads
        keyboard_thread.start()
        camera_thread.start()
        processing_thread.start()
        save_thread.start()
        
        # Set CPU affinities for better performance on Jetson Nano
        set_thread_affinity(keyboard_thread, [0])
        set_thread_affinity(camera_thread, [1])
        set_thread_affinity(processing_thread, [2])
        set_thread_affinity(save_thread, [3])
        
        print(f"Recording started. Press Q to stop.")
        start_time = time.time()
        
        try:
            while not self.terminate and (time.time() - start_time < duration):
                time.sleep(0.1)  # Main thread just monitors the others
                
        finally:
            self.terminate = True
            self.pipeline.stop()
            if self.ser:
                self.ser.close()
            if self.h5_file:
                self.h5_file.close()
            
            # Wait for threads to finish
            keyboard_thread.join()
            camera_thread.join()
            processing_thread.join()
            save_thread.join()
            
            print(f"Recording complete. Saved {self.sample_count} samples")

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.capture_data(duration=np.inf)