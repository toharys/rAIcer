import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robot_control import Action, execute_action
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

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
        self.render_initialized = False
        # todo: what is the last_action initial value
        self.last_action
        self.score = 0
        self.history = []

    def reset(self, seed=None, options=None):
        """ Reset RaicerEnv to initial state."""
        self.state = INITIAL_STATE
        self.steps = 0
        self.score = 0
        # todo: what is the last_action initial value
        self.last_action = None
        self.history = []
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
        self.steps += 1

        # Termination conditions
        reward, done = self.reward()
        self.score += reward
        self.last_action = action
        truncated = self.steps >= self.max_steps  # Max steps reached

        return self.state, reward, done, truncated, {}

    def render(self, mode="human"):
        if not self.render_initialized:
            plt.ion()   # interactive mode on - enable to keep update without blocking the rest code
            self.fig, self.ax = plt.subplots(figsize=(6,6))
            self.render_initialized = True
        self.ax.clear()

        # Track position
        x, y, direction = self.state
        self.history.append(((x,y)))

        # Plot path history
        if len(self.history)>1:
            xs, ys = zip(*self.history)     # set of the x's and set of the y's
            self.ax.plot(xs, ys, 'gray', linestyle='--', alpha=0.5)

        # Plot robot as arrow
        arrow = FancyArrowPatch((x, y),
                                (x + 3 * np.cos(np.deg2rad(direction)),
                                 y + 3 * np.sin(np.deg2rad(direction))),
                                color='blue', arrowstyle='->', mutation_scale=10)
        self.ax.add_patch(arrow)

        # Plot goal
        goal_x, goal_y = self.goal
        self.ax.plot(goal_x, goal_y, 'ro', label='Goal')

        # Styling
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("RaicerEnv Simulation")
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.legend()
        plt.pause(0.01)

        # todo: add displaying the score and last actione




