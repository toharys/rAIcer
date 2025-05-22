import gymnasium as gym
import numpy as np
import time

# Create the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)

# Reset the environment
obs, _ = env.reset()

# Define a constant forward action:
# [steering, gas, brake]
# steering = 0 (no turn), gas = 1 (full throttle), brake = 0
forward_action = np.array([0.0, 1.0, 0.0])

# Run the simulation
for _ in range(1000):  # number of steps
    obs, reward, done, truncated, info = env.step(forward_action)
    env.render()
    time.sleep(0.01)  # Slow it down a bit for visibility
    if done or truncated:
        break

env.close()
