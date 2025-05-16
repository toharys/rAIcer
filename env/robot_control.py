import serial
import glob
import os
import threading
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
from enum import Enum
from evdev import InputDevice, categorize, ecodes

# Path to keyboard input event
KEYBOARD_PATH = "/dev/input/event2"  # Adjust based on `cat /proc/bus/input/devices`

# Define action enumeration
class Action(Enum):
    STOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    BACKWARD = 4

class Robot:
    def __init__(self):

        self.keymap = {
            "KEY_RIGHT": b'w',
            "KEY_LEFT": b's',
            "KEY_UP": b'u',
            "KEY_DOWN": b'b',
            "KEY_SPACE": b'x'
        }

        self.connect_to_arduino()
        
        self.connect_to_keyboard()
        
        self.camera_ready = False
        self.initialize_camera()
        
        self.serial_lock = threading.Lock()
        self.feedback_thread = threading.Thread(target=self.read_arduino, daemon=True)
        self.feedback_thread.start()
        
    
    def connect_to_arduino(self):
        """Connect to the Arduino via USB serial port"""
        serial_ports = glob.glob('/dev/serial/by-id/*')
        if len(serial_ports) == 0:
            raise Exception("No serial device found. Check the connection.")
        
        # Resolve the symbolic link to get the actual port path
        self.arduino_port = os.path.realpath(serial_ports[0])
        self.baud_rate = 9600
        
        try:
            self.ser = serial.Serial(self.arduino_port, self.baud_rate, timeout=1)
            print(f"Connected to Arduino on {self.arduino_port}")
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")
            exit(1)
    
    def connect_to_keyboard(self):
        """Connect to the keyboard input device"""
        try:
            self.keyboard = InputDevice(KEYBOARD_PATH)
            print(f"Listening for key presses on {self.keyboard.path}...")
        except Exception as e:
            print(f"Error connecting to keyboard: {e}")
            exit(1)
    
    def initialize_camera(self):
        """Initialize the RealSense depth camera"""
        try:
            self.pipe = rs.pipeline()
            self.cfg = rs.config()
            
            # Configure streams
            self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start streaming
            self.pipe.start(self.cfg)
            self.camera_ready = True
            print("RealSense camera initialized successfully")
        except Exception as e:
            print(f"Error initializing RealSense camera: {e}")
            self.pipe = None
    
    def get_grayscale_frame(self):
        """Get a grayscale frame from the RealSense camera
        
        Returns:
            torch.Tensor: Grayscale image as a PyTorch tensor
        """
        if not self.camera_ready:
            print("Camera is not ready")
            return None
        
        try:
            # Wait for a coherent pair of frames
            frames = self.pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("Failed to get color frame")
                return None
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert to grayscale
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Convert to PyTorch tensor
            # Normalize to [0, 1] range
            tensor = torch.from_numpy(gray_image.astype(np.float32) / 255.0)
            
            # Add batch and channel dimensions [H, W] -> [1, 1, H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            
            return tensor
        except Exception as e:
            print(f"Error getting grayscale frame: {e}")
            return None
    
    def get_depth_frame(self):
        """Get a depth frame from the RealSense camera
        
        Returns:
            torch.Tensor: Raw depth image as a PyTorch tensor
        """
        if self.pipe is None:
            print("Camera is not initialized")
            return None
        
        try:
            # Wait for a coherent pair of frames
            frames = self.pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if not depth_frame:
                print("Failed to get depth frame")
                return None
            
            # Convert to numpy array - raw depth values
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert to PyTorch tensor
            # The depth values are in millimeters, so we normalize by dividing by 65535
            # to get a range of [0, 1] for most typical depths
            tensor = torch.from_numpy(depth_image.astype(np.float32) / 65535.0)
            
            # Add batch and channel dimensions [H, W] -> [1, 1, H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            
            return tensor
        except Exception as e:
            print(f"Error getting depth frame: {e}")
            return None
    
    def get_colorized_depth_frame(self):
        """Get a colorized depth frame from the RealSense camera
        
        Returns:
            torch.Tensor: Colorized depth image as a PyTorch tensor (using JET colormap)
        """
        if self.pipe is None:
            print("Camera is not initialized")
            return None
        
        try:
            # Wait for a coherent pair of frames
            frames = self.pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if not depth_frame:
                print("Failed to get depth frame")
                return None
            
            # Convert to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Apply colormap
            depth_colorized = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.5),
                cv2.COLORMAP_JET
            )
            
            # Convert to PyTorch tensor
            # OpenCV uses BGR order, convert to RGB
            depth_colorized_rgb = cv2.cvtColor(depth_colorized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] range
            tensor = torch.from_numpy(depth_colorized_rgb.astype(np.float32) / 255.0)
            
            # Permute dimensions to get [C, H, W] for PyTorch convention
            tensor = tensor.permute(2, 0, 1)
            
            # Add batch dimension [C, H, W] -> [1, C, H, W]
            tensor = tensor.unsqueeze(0)
            
            return tensor
        except Exception as e:
            print(f"Error getting colorized depth frame: {e}")
            return None
    
    def read_arduino(self):
        """Continuously read messages from Arduino"""
        while True:
            try:
                if self.ser.in_waiting > 0:
                    with self.serial_lock:
                        data = self.ser.readline().decode('utf-8').strip()
                        if data:
                            print(f"Arduino: {data}")
            except Exception as e:
                print(f"Error reading from Arduino: {e}")

        
    def start_control(self):
        """Listen for keyboard inputs and send commands to the Arduino"""
        print("Starting keyboard control. Use arrow keys and space to control the robot.")
        
        for event in self.keyboard.read_loop():
            if event.type == ecodes.EV_KEY and event.value == 1:  # Key press event
                key_event = categorize(event)
                
                # Send command to Arduino based on key press
                keycode = key_event.keycode
                if keycode in self.keymap:
                    with self.serial_lock:
                        self.ser.write(self.keymap[keycode])
                        print(f"Sending: {self.keymap[keycode].decode()} ({keycode})")

    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'ser'):
            self.ser.close()
            print("Serial connection closed")
        
        if self.camera_ready:
            self.pipe.stop()
            print("Camera stopped")


# if __name__ == "__main__":
#     # Create robot instance
#     robot = Robot()
    
#     # Start control
#     try:
#         robot.start_control()
#     except KeyboardInterrupt:
#         print("\nExiting program")
#     finally:
#         robot.close()