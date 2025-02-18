import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robot_control import Action, execute_action

# todo: ~~~~~ DECIDE WHAT IS THE INITIAL POINT ~~~~~
INITIAL_STATE: np.ndarray = np.array([50, 50, 0])
# todo: ~~~~~ DECIDE WHAT IS THE FINAL GOAL POSITION ~~~~~
GOAL_POS: np.ndarray = np.array([90, 90])
# todo: ~~~~~ DECIDE WHAT IS THE MAX STEPS NUM~~~~~
MAX_STEPS_NUM: int = 1000

class RaicerEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Action space (0=do nothing(stop), 1=steer left, 2=steer right, 3=forward, 4=backward)
        self.action_space = spaces.Discrete(5)

        # Observation_space (x,y, direction)
        # x,y: position in a 2D plane
        # direction: angle in degrees (0-360) measured relative to the right "x-axis"
        # todo: ~~~~~ DECIDE WHAT ARE THE RIGHT SIZES ~~~~~
        self.observation_space = spaces.Box(
            low=np.array([0,0,0]),
            high=np.array([100,100,360]),
            dtype=np.float32
        )

        # Initialize state (x, y, direction)
        # todo: ~~~~~ DECIDE WHAT IS THE INITIAL POINT ~~~~~
        self.state = INITIAL_STATE

        # Define the goal position
        # todo: ~~~~~ DECIDE WHAT IS THE FINAL GOAL POINT ~~~~~
        self.goal = GOAL_POS

        # Define maximum number of steps before termination
        # todo: ~~~~~ DECIDE WHAT IS THE MAX STEPS NUM~~~~~
        self.max_steps = MAX_STEPS_NUM
        self.steps = 0

        def reset(self, seed=None, options=None):
            """ Reset RaicerEnv to initial state."""
            self.state = INITIAL_STATE
            self.steps = 0
            return self.state, {}

        def reward(self):
            reward = 0
            # penalize distance from the final position
            distance_to_goal = np.linalg.norm(self.state[:2]-self.goal)
            reward -= distance_to_goal / 100
            # todo: collision
            if(False):
                reward -= 50
            # todo: within the borders
            if(False):
                reward += 5
            done = distance_to_goal < 5  # Reached goal
            return reward, done

        def step(self, action):
            """ Apply action and return new state, reward, done, truncated, info."""
            x, y, direction = self.state
            x, y, direction = execute_action(x, y, direction, action)
            self.state = np.array([x, y, direction])

            # Termination conditions
            reward, done = self.reward()
            truncated = self.steps >= self.max_steps  # Max steps reached

            return self.state, reward, done, truncated, {}



