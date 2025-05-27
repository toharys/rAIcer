import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
from rAIcer.robot_control import Robot, FrameStack
import time

def capture_frame_stacks(num_stacks=100, stack_size=4, frame_type='grayscale'):
    """
    Capture N frame stacks from RealSense camera
    
    Args:
        num_stacks: Number of frame stacks to capture
        stack_size: Number of frames per stack
        frame_type: 'grayscale' or 'depth' or 'color'
    
    Returns:
        List of (frame_stack, action) tuples
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)
    frame_stack = FrameStack(stack_size)
    recorded_stacks = []
    
    try:
        print(f"Capturing {num_stacks} frame stacks (size={stack_size})...")
        
        # Initialize with first frame
        frames = pipeline.wait_for_frames()
        first_frame = process_frame(frames, frame_type)
        if first_frame is None:
            raise ValueError("Failed to get initial frame")
        frame_stack.reset(first_frame)
        
        while len(recorded_stacks) < num_stacks:
            # Get frames
            frames = pipeline.wait_for_frames()
            current_frame = process_frame(frames, frame_type)
            
            if current_frame is None:
                continue
            
            # Get the current frame stack
            stacked_frames = frame_stack.step(current_frame)
            
            # Record the stack (with default 'stop' action)
            recorded_stacks.append((stacked_frames, 'stop'))
            
            # Display (optional)
            display_frame = current_frame if current_frame.ndim == 3 else cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Current Frame', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            print(f"Captured {len(recorded_stacks)}/{num_stacks}", end='\r')
            
            # Add 50ms delay between capturing stacks <-- NEW
            time.sleep(0.05)  # Critical for timing control
        
        print(f"\nSuccessfully captured {len(recorded_stacks)} frame stacks")
        return recorded_stacks
        
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def process_frame(frames, frame_type):
    """Process frames based on requested type"""
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        return None
    
    if frame_type == 'grayscale':
        color_image = np.asanyarray(color_frame.get_data())
        return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    elif frame_type == 'depth':
        depth_image = np.asanyarray(depth_frame.get_data())
        return cv2.convertScaleAbs(depth_image, alpha=0.03)
    elif frame_type == 'color':
        return np.asanyarray(color_frame.get_data())
    else:
        raise ValueError(f"Unknown frame type: {frame_type}")

def save_frame_stacks(stacks, filename):
    """Save frame stacks to NPZ file"""
    np.savez_compressed(
        filename,
        stacks=[s[0] for s in stacks],  # Frame data
        actions=[s[1] for s in stacks]   # Corresponding actions
    )
    print(f"Saved {len(stacks)} stacks to {filename}")

def load_frame_stacks(filename):
    """Load saved frame stacks"""
    data = np.load(filename)
    return list(zip(data['stacks'], data['actions']))

def visualize_stack(stack):
    """Visualize a frame stack"""
    frames, action = stack
    num_frames = frames.shape[0]
    
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    if num_frames == 1:
        axes = [axes]
    
    for i in range(num_frames):
        if frames[i].ndim == 2:  # Grayscale/depth
            axes[i].imshow(frames[i], cmap='gray')
        else:  # Color
            axes[i].imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis('off')
    
    plt.suptitle(f"Action: {action}")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Capture 50 grayscale frame stacks with 4 frames each
    stacks = capture_frame_stacks(num_stacks=50, stack_size=4, frame_type='grayscale')
    
    # Save to file
    if stacks:
        save_frame_stacks(stacks, "grayscale_stacks.npz")
        
        # Visualize the first stack
        visualize_stack(stacks[0])
        
        # Example of loading saved stacks
        loaded_stacks = load_frame_stacks("grayscale_stacks.npz")
        print(f"Loaded {len(loaded_stacks)} stacks")

