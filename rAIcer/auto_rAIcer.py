import os
import torch
from agent import rAIcerAgent
from robot_control import Robot, Action
import time


AGENT_PATH = os.path.join("./models", "trained_agent.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# Load the agent
checkpoint = torch.load(AGENT_PATH, map_location=DEVICE)
in_channels = 4     # 4 binary frames
num_actions = 5

agent = rAIcerAgent(state_shape=(in_channels, 120, 160), num_actions=num_actions)  # Adjust H, W if needed
agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
agent.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
agent.step_counter = checkpoint.get('step_counter', 0)
agent.q_network.to(DEVICE)
agent.target_q_network.to(DEVICE)
agent.q_network.eval()
agent.target_q_network.eval()

# Initialize robot
robot = Robot()

print("Robot initialized. Starting control loop...")

try:
    while not robot.terminate:
        stacked = robot.get_stacked_frames()
        if stacked is None:
            continue

        state = torch.from_numpy(stacked).float().unsqueeze(0).to(DEVICE)
        action_idx = agent.select_action(state, mode="mean")
        action = Action(action_idx)

        robot.control_action(action)
        time.sleep(robot.calculate_sample_interval())

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    robot.close()
    print("Shutdown complete")

