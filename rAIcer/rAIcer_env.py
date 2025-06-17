import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
# from robot_control import Action
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch


class Action(Enum):
    STOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    BACKWARD = 4

# JUST NEED TO DEFINE THE REWARD FUNCTION

# todo: ~~~~~ DECIDE WHAT IS THE INITIAL POINT ~~~~~
INITIAL_STATE: np.ndarray = np.array([50, 50, 0])
# todo: ~~~~~ DECIDE WHAT IS THE FINAL GOAL POSITION ~~~~~
GOAL_POS: np.ndarray = np.array([90, 90])
# todo: ~~~~~ DECIDE WHAT IS THE MAX STEPS NUM~~~~~
MAX_STEPS_NUM: int = 1000


# class RaicerEnv(gym.Env):
#     def __init__(self):
#         super().__init__()
#
#         # Action space (0=do nothing(stop), 1=steer left, 2=steer right, 3=forward, 4=backward)
#         self.action_space = spaces.Discrete(5)
#
#         # Observation_space (x,y, direction)
#         # x,y: position in a 2D plane
#         # direction: angle in degrees (0-360) measured relative to the right "x-axis"
#         # todo: ~~~~~ DECIDE WHAT ARE THE RIGHT SIZES ~~~~~
#         self.observation_space = spaces.Box(
#             low=np.array([0,0,0]),
#             high=np.array([100,100,360]),
#             dtype=np.float32
#         )
#
#         # Initialize state (x, y, direction)
#         # todo: ~~~~~ DECIDE WHAT IS THE INITIAL POINT ~~~~~
#         self.state = INITIAL_STATE
#
#         # Define the goal position
#         # todo: ~~~~~ DECIDE WHAT IS THE FINAL GOAL POINT ~~~~~
#         self.goal = GOAL_POS
#
#         # Define maximum number of steps before termination
#         # todo: ~~~~~ DECIDE WHAT IS THE MAX STEPS NUM~~~~~
#         self.max_steps = MAX_STEPS_NUM
#         self.steps = 0
#         self.render_initialized = False
#         # todo: what is the last_action initial value
#         self.last_action
#         self.score = 0
#         self.history = []
#
#     def reset(self, seed=None, options=None):
#         """ Reset RaicerEnv to initial state."""
#         self.state = INITIAL_STATE
#         self.steps = 0
#         self.score = 0
#         # todo: what is the last_action initial value
#         self.last_action = None
#         self.history = []
#         return self.state, {}
#
#     def reward(self):
#         reward = 0
#         # penalize distance from the final position
#         distance_to_goal = np.linalg.norm(self.state[:2]-self.goal)
#         reward -= distance_to_goal / 100
#         # todo: collision
#         if(False):
#             reward -= 50
#         # todo: within the borders
#         if(False):
#             reward += 5
#         done = distance_to_goal < 5  # Reached goal
#         return reward, done
#
#     def step(self, action):
#         """ Apply action and return new state, reward, done, truncated, info."""
#         x, y, direction = self.state
#         x, y, direction = execute_action(x, y, direction, action)
#         self.state = np.array([x, y, direction])
#         self.steps += 1
#
#         # Termination conditions
#         reward, done = self.reward()
#         self.score += reward
#         self.last_action = action
#         truncated = self.steps >= self.max_steps  # Max steps reached
#
#         return self.state, reward, done, truncated, {}
#
#     def render(self, mode="human"):
#         if not self.render_initialized:
#             plt.ion()   # interactive mode on - enable to keep update without blocking the rest code
#             self.fig, self.ax = plt.subplots(figsize=(6,6))
#             self.render_initialized = True
#         self.ax.clear()
#
#         # Track position
#         x, y, direction = self.state
#         self.history.append(((x,y)))
#
#         # Plot path history
#         if len(self.history)>1:
#             xs, ys = zip(*self.history)     # set of the x's and set of the y's
#             self.ax.plot(xs, ys, 'gray', linestyle='--', alpha=0.5)
#
#         # Plot robot as arrow
#         arrow = FancyArrowPatch((x, y),
#                                 (x + 3 * np.cos(np.deg2rad(direction)),
#                                  y + 3 * np.sin(np.deg2rad(direction))),
#                                 color='blue', arrowstyle='->', mutation_scale=10)
#         self.ax.add_patch(arrow)
#
#         # Plot goal
#         goal_x, goal_y = self.goal
#         self.ax.plot(goal_x, goal_y, 'ro', label='Goal')
#
#         # Styling
#         self.ax.set_xlim(0, 100)
#         self.ax.set_ylim(0, 100)
#         self.ax.set_title("RaicerEnv Simulation")
#         self.ax.set_aspect('equal')
#         self.ax.grid(True)
#         self.ax.legend()
#         plt.pause(0.01)
#

def center_score(frame: np.ndarray) -> float:
    """
    penalize proportional to the deviatation of the white line from the frame center
    :param frame:
    :return:
    """
    h, w = frame.shape
    center_x = w // 2
    row_y = int(0.75 * h)   # horizontal row of pixels located at 75% of the image height
    row = frame[row_y, :]

    white_pixels = np.where(row == 255)[0]
    if len(white_pixels) == 0:
        return 1.0  # Max penalty if no line detected

    # line_center = int(np.mean(white_pixels))
    line_center = int(np.mean(white_pixels).item())
    offset = abs(center_x - line_center)
    return offset / center_x    # Normalize 0 = best, 1 = worst

def continuity_score(frame: np.ndarray) -> float:
    """
    Reward based on how continuous the white path is (fewer black gaps inside).
    """
    row_y = int(0.75 * frame.shape[0])
    row = frame[row_y, :]
    white_pixels = (row == 255).astype(np.uint8)
    transitions = np.count_nonzero(np.diff(white_pixels))  # Count changes between black and white
    # Normalize: fewer transitions (clean line) = better
    return 1.0 - min(transitions / frame.shape[1], 1.0)

def frame_difference(f1: np.ndarray, f2: np.ndarray) -> float:
    diff = np.abs(f1.astype(np.int16) - f2.astype(np.int16))  # Avoid overflow
    return np.mean(diff) / 255.0  # Normalized difference (0 to 1)

def compute_reward(frames: np.ndarray, current_action: Action, previous_action: Action) -> tuple[float, bool]:
    """
      Compute reward for line-following.
      - Encourages staying centered.
      - Penalizes deviation and lateral shifts.
      - Penalizes STOP and frequent action changes.
    """
    reward = 0.0
    center_scores = []
    continuity_scores = []

    for frame in frames:
        c_score = center_score(frame)   # 0 (perfect center) - 1 (edge)
        cont_score = continuity_score(frame)

        center_scores.append(c_score)
        continuity_scores.append(cont_score)

        if current_action != Action.STOP:
            reward += 1
        else:
            reward -= 2

        reward += 1.0 * cont_score
        reward += 1.0 * (1.0 - c_score)     # +1 if perfectly centered, down to 0

    reward /= len(frames)   # average over frames
    # reward -= 0.5   # small time penalty

    # Temporal smoothness
    no_movement_penalize = 0.0

    for t in range(1, len(center_scores)):
        diff = abs(center_scores[t] - center_scores[t-1])
        no_movement_penalize += frame_difference(frames[t], frames[t-1])
        reward -= 0.2 * diff    # penalize sudden lateral shifts
    if no_movement_penalize >= 0.5:
        reward -= 0.7

    # Penalize action changes
    if previous_action is not None and current_action != previous_action:
        reward -= 0.3

    # Penalize stopping if applicable
    if current_action == Action.STOP:
        reward -= 0.5   # its common to stop between action to action so moderate panelize

    # Bonus if continuity and center are both very good
    if all(cs < 0.2 for cs in center_scores) and all(s > 0.9 for s in continuity_scores):
        reward += 0.5  # small positive boost

    return float(reward), False #float(np.clip(reward, -2.0, 2.0)), False








# Two lines borders - REMOVED

# FINAL_SIGN_TEMPLATE: torch.Tensor = cv2.imread("path/to/final_sign_template.png", cv2.IMREAD_GRAYSCALE)     # should be a the halt frame with the sign that will appear in the frame
#
# def scan_final_sign(frame: np.ndarray) -> tuple[float, bool]:
#     """
#     recieves frame and return a score proportional to fraction that the final sign take within the frame
#     :param frame:
#     :return: (reward, done)
#     """
#     global FINAL_SIGN_TEMPLATE
#     if FINAL_SIGN_TEMPLATE is None:
#         return 0.0, False  # No template available
#
#     # ORB detector
#     orb = cv2.ORB_create()
#
#     # Detect keypoints and descriptors
#     kp1, des1 = orb.detectAndCompute(FINAL_SIGN_TEMPLATE, None)
#     kp2, des2 = orb.detectAndCompute(frame, None)
#
#     if des1 is None or des2 is None:
#         return 0.0, False
#
#     # Brute-force matcher with Hamming distance (since ORB uses binary descriptors)
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#
#     if not matches:
#         return 0.0, False
#
#     # Sort matches by distance (lower=better)
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     # Normalize score based on match quality
#     good_matches = [m for m in matches if m.distance < 64]  # todo: how to tune the threshold??
#     score = len(good_matches) / max(len(kp1), 1)    # how much from the final sign keypoints appears in the frame (matching condidence)
#     done = score >= 0.8
#     # Clamp to [0,1]
#     return min(score, 1.0), done
#
# def edge_distances_from_center(frame: np.ndarray) -> tuple[int, int]:
#     h, w = frame.shape
#     center_x = w//2
#     row_y = int(0.75*h)     # pick a row near the bottom
#     row = frame[row_y, :]
#
#     # finds the indices of white
#     white_pixels = np.where(row == 255)[0]
#     if len(white_pixels) < 2:
#         return None, None   # Not enough info
#
#     left_border = white_pixels[0]
#     right_border = white_pixels[-1]
#
#     dist_to_left = center_x - left_border
#     dist_to_right = right_border - center_x
#
#     return dist_to_left, dist_to_right
#
# def compute_reward(frames: np.ndarray, current_action: Action, previous_action: Action) -> tuple[float, bool]:
#     """
#     given 4 frames stack return the current reward and weather we done or not
#     :param frames:
#     :return:
#     """
#     reward = 0
#     last_done, second_last_done = False, False
#     # Per-frame rewards
#     for frame in frames:
#         final_sign_score, _ = scan_final_sign(frame)
#         reward += 2.0 * final_sign_score
#         reward -= 0.5 * center_score(frame)
#         reward -= 1.0 * edge_penalty(frame)
#     reward -= 1.0  # time penalty
#
#     # Temporal (cross-frame) rewards
#     for t in range(1, len(frames)):
#         final_sign_score_t, done_t = scan_final_sign(frames[t])
#         final_sign_score_prev, done_prev = scan_final_sign(frames[t-1])
#
#         if t == len(frames)-1:
#             last_done, second_last_done = done_t, done_prev
#
#         delta = final_sign_score_t - final_sign_score_prev
#         reward += 4.0 * np.tanh(delta * 5)  # non linear in delta, reward sign proximity more heavily
#         lateral_diff = center_score(frames[t]) - center_score(frames[t-1])
#         reward -= 0.3 * abs(lateral_diff)
#
#     # Action smoothness penalty
#     if previous_action is not None and current_action != previous_action:
#         reward -= 1.0
#
#     # _, done = scan_final_sign(frames[-1])
#
#     return reward, (last_done and second_last_done)
#
# def center_score(frame: np.ndarray) -> float:
#     """
#     should penalize proportional to devistate from the center
#     :param frame:
#     :return:
#     """
#     distance_from_left_border, distance_from_right_border = edge_distances_from_center(frame)
#     if distance_from_left_border is None or distance_from_right_border is None:
#         return 10.0     # todo: to check if it is large enough
#     return abs(distance_from_right_border-distance_from_left_border)
#
# def edge_penalty(frame: np.ndarray) -> float:
#     """
#     Penalizes situations where the robot drifts too far left or right (i.e., sees less road).
#     :param frame:
#     :return:
#     """
#     left, right = edge_distances_from_center(frame)
#     if left is None or right is None:
#         return 1.0  # maximum penalty if we can't detect borders
#
#     road_width = left + right
#     expected_width = frame.shape[1] * 0.8  # expected visible road span (tunable)
#
#     # Penalize if the road width is too narrow
#     penalty = max(0.0, (expected_width - road_width) / expected_width)
#     return penalty
