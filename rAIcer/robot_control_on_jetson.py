import os
import serial
import glob
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque
import queue
import psutil
from enum import Enum
import socket
import pickle
import struct
import subprocess

def get_ip_address():
    """Get IP address without netifaces dependency"""
    try:
        # Try getting IP via hostname
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith('127.'):
            return ip
        
        # Fallback to using system commands
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.split()[0]
    except:
        pass
    return '0.0.0.0'

# Configuration
ARDUINO_PORTS = glob.glob('/dev/serial/by-id/*')
FRAME_TYPE = 'grayscale'  # 'grayscale', 'depth', or 'color'
STACK_SIZE = 4
MIN_SAMPLE_INTERVAL = 0.05  # 20Hz max sample rate
NYQUIST_MULTIPLIER = 2.5   # Sample 2.5x faster than action changes
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 3200

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

class RobotServer:
    def __init__(self):
        self.actual_ip = get_ip_address()
        print(f"Jetson actual IP: {self.actual_ip}")
        self.ser = None
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
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        
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
        """Connect to Arduino"""
        if not ARDUINO_PORTS:
            raise Exception("No Arduino found")
       
        arduino_port = os.path.realpath(ARDUINO_PORTS[0])
        try:
            self.ser = serial.Serial(arduino_port, 9600, timeout=0.1, write_timeout=0.1)
            print(f"Connected to Arduino on {arduino_port}")
        except Exception as e:
            print(f"Arduino error: {e}")
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

    def control_action(self, action):
        """Send command to Arduino based on action"""
        if action in self.keymap:
            self.current_action = action
            self.ser.write(self.keymap[action])
            print(f"Sending: {self.keymap[action].decode()} ({action})")

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
    
    def send_stacked_frames(self, conn, stacked_frames):
        """Send properly formatted stacked frames"""
        if stacked_frames is None:
            return
        
        try:
            # Convert frames to list of bytes
            frames_data = []
            for i in range(stacked_frames.shape[0]):
                frame = stacked_frames[i]
                frame = cv2.resize(frame, (160, 120))  # Resize to expected dimensions
                _, buffer = cv2.imencode('.jpg', frame)
                frames_data.append(buffer.tobytes())
                
            # Create simple list structure (not dict)
            data = pickle.dumps(frames_data)
            conn.sendall(struct.pack(">L", len(data)) + data)
             
        except Exception as e:
            print(f"Error sending frames: {e}")

    def handle_client(self, conn):
        """Handle client connection"""
        try:
            while not self.terminate:
                # Receive command from client
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    
                    command = data.decode().strip()
                    if command.startswith("ACTION:"):
                        action_name = command.split(":")[1]
                        try:
                            action = Action[action_name]
                            self.control_action(action)
                        except KeyError:
                            print(f"Unknown action: {action_name}")
                    elif command == "GET_STACKED_FRAMES":
                        stacked = self.get_stacked_frames()
                        self.send_stacked_frames(conn, stacked)
                except ConnectionResetError:
                    break
                except Exception as e:
                    print(f"Error handling client: {e}")
                    break
        finally:
            conn.close()


    def run_server(self):
        """Run the server with better error handling and diagnostics"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.settimeout(5.0)  # Add timeout to accept()
                  
                try:
                    s.bind((SERVER_HOST, SERVER_PORT))
                except OSError as e:
                    print(f"Bind failed: {e}")
                    print(f"Port {SERVER_PORT} may be in use")
                    print("Try: sudo netstat -tulnp | grep {SERVER_PORT}")
                    self.terminate = True
                    return
                    
                s.listen(5)
                print(f"\n=== Server ACTUALLY listening on ===")
                print(f"All interfaces (0.0.0.0):{SERVER_PORT}")
                print(f"Specific IP ({self.actual_ip}):{SERVER_PORT}\n")
                
                while not self.terminate:
                    try:
                        print("[Server] Waiting for connection...")
                        conn, addr = s.accept()
                        print(f"[Server] Connection from {addr}")
                          
                        # Set socket timeouts
                        conn.settimeout(10.0)
                        
                        # Handle client in main thread
                        self.handle_client(conn)
                        
                    except socket.timeout:
                        print("[Server] Accept timeout (normal while waiting)")
                        continue
                    except Exception as e:
                        print(f"[Server] Connection error: {e}")
                        if 'conn' in locals():
                            conn.close()
                        continue
                        
            except Exception as e:
                print(f"[Server] FATAL ERROR: {e}")
            finally:
                print("Server shutdown complete")

     
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
        
        if self.server_thread.is_alive():
            self.server_thread.join()

if __name__ == "__main__":
    server = RobotServer()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.close()
        print("Server shutdown complete")