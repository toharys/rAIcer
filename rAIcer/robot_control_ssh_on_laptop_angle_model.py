import socket
import cv2
import numpy as np
import pickle
import struct
from enum import Enum
import time

FRAME_TYPE = 'grayscale'
STACK_SIZE = 4

class Action(Enum):
    STOP = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4

class Robot:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.TARGET_FPS = 30
        self.SAMPLE_INTERVAL = 1.0 / self.TARGET_FPS  # 0.033s
        self.last_sample_time = time.time()
        self.socket = None
        self.terminate = False
        self.action_durations = []
        self.last_action_change_time = time.time()
        self.last_sample_time = time.time()
        self.connect()

    def connect(self):
        """Connect to the Jetson server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            print(f"Connected to server at {self.host}:{self.port}")
        except Exception as e:
            print(f"Connection error: {e}")
            raise
    
    def control_action(self, action):
        """Send an action command to the server"""
        if not isinstance(action, Action):
            raise ValueError("action must be an Action enum")
        
        try:
            self.socket.sendall(f"ACTION:{action.name}".encode())
            now = time.time()
            self.action_durations.append(now - self.last_action_change_time)
            self.last_action_change_time = now
        except Exception as e:
            print(f"Error sending action: {e}")
            self.reconnect()

    def get_stacked_frames(self):
        """Request and receive frames at exactly 30Hz (33.3ms intervals)"""
        # 1. Enforce timing
        now = time.time()
        elapsed = now - self.last_sample_time
        if elapsed < self.SAMPLE_INTERVAL:
            time.sleep(self.SAMPLE_INTERVAL - elapsed)

        # 2. Request frames
        try:
            self.socket.sendall(b"GET_STACKED_FRAMES")

            # 3. Get frame data size
            size_data = self._recvall(4)
            if not size_data:
                return None
            size = struct.unpack(">L", size_data)[0]

            # 4. Receive serialized data
            serialized = self._recvall(size)
            if not serialized:
                return None

            # 5. Decode frames
            frames_data = pickle.loads(serialized)
            frames = []

            for frame_bytes in frames_data:
                frame = cv2.imdecode(
                    np.frombuffer(frame_bytes, dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE if FRAME_TYPE == 'grayscale' else cv2.IMREAD_COLOR
                )
                if frame is not None:
                    frames.append(frame)

            # 6. Update timing and return
            self.last_sample_time = time.time()

            if len(frames) == STACK_SIZE:
                return np.stack(frames, axis=0)  # Shape: [4, H, W]

        except Exception as e:
            print(f"Frame receive error: {e}")
            self.reconnect()

        return None
    
    def get_state(self):
        """Get complete state from server (frames, angle, prev_action)"""
        # Enforce timing
        now = time.time()
        elapsed = now - self.last_sample_time
        if elapsed < self.SAMPLE_INTERVAL:
            time.sleep(self.SAMPLE_INTERVAL - elapsed)

        # Request state
        try:
            self.socket.sendall(b"GET_STATE")

            # Get data size
            size_data = self._recvall(4)
            if not size_data:
                return None
            size = struct.unpack(">L", size_data)[0]

            # Receive serialized data
            serialized = self._recvall(size)
            if not serialized:
                return None

            # Decode state
            state = pickle.loads(serialized)
            
            # Update timing and return
            self.last_sample_time = time.time()
            return state

        except Exception as e:
            print(f"State receive error: {e}")
            self.reconnect()
            return None
    
    def calculate_sample_interval(self):
       return self.SAMPLE_INTERVAL 
        
    def _recvall(self, count):
        """Helper function to receive all data"""
        buf = b''
        while count:
            newbuf = self.socket.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    
    def reconnect(self):
        """Attempt to reconnect to the server"""
        self.close()
        time.sleep(1)
        self.connect()
    
    def close(self):
        """Close the connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.terminate = True