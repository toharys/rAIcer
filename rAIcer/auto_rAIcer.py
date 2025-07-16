
import os
import numpy as np
import torch
from agent import rAIcerAgent
#from robot_control import Robot, Action
#from robot_control_ssh_on_laptop import Robot, Action
from robot_control_ssh_on_laptop_angle_model import Robot, Action
from behavior_cloning_policy import BehaviorCloningPolicy
import time

BC_PATH = os.path.join("rAIcer\models", "bc_model_25_6.pt")
AGENT_PATH = os.path.join("rAIcer\models", "trained_agent.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Constants (tune these as needed)
DEVIATION_THRESHOLD = 40  # pixels
HEURISTIC_ADJUSTMENT = 5.0  # degrees to shift angle
IMAGE_WHITE_THRESHOLD = 200  # pixel value considered "white"

def correction_heuristic(stacked_frames, angle, prev_action):
    """
    Applies a heuristic servo correction based on white line location
    
    Args:
        bc_model: trained behavior cloning model
        stacked_frames: Tensor [4, H, W], image stack
        angle: float (servo angle)
        prev_action: int (previous action taken)
        device: torch device

    Returns:
        action: int (predicted action)
    """
    action = -1
    # Heuristic: check the last frame for white line deviation
    last_frame = stacked_frames[-1] #.cpu().numpy()  # [H, W]
    white_pixels = np.where(last_frame > IMAGE_WHITE_THRESHOLD)
    
    if len(white_pixels[1]) > 0:
        avg_x = np.mean(white_pixels[1])
        frame_center = last_frame.shape[1] / 2
        deviation = avg_x - frame_center

        if deviation > DEVIATION_THRESHOLD:
            action = 3
        elif deviation < -DEVIATION_THRESHOLD:
            action = 4


    return action

# Load the agent
# checkpoint = torch.load(AGENT_PATH, map_location=DEVICE)
in_channels = 4     # 4 binary frames
num_actions = 5

# agent = rAIcerAgent(state_shape=(in_channels, 120, 160), num_actions=num_actions)  # Adjust H, W if needed
# agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
# agent.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
# agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# agent.step_counter = checkpoint.get('step_counter', 0)
# agent.q_network.to(DEVICE)
# agent.target_q_network.to(DEVICE)
# agent.q_network.eval()
# agent.target_q_network.eval()
bc_model =  BehaviorCloningPolicy(in_channels=in_channels, num_actions=num_actions)
bc_model.load_state_dict(torch.load(BC_PATH, map_location=DEVICE)) #, map_location=DEVICE)
bc_model.eval()

# Initialize robot
#robot = Robot()
robot = Robot('10.100.102.26' , 3200)

print("Robot initialized. Starting control loop...")

try:
    while not robot.terminate:
        #stacked = robot.get_stacked_frames()
        state = robot.get_state()
        if state is None:
            continue

        #state = torch.from_numpy(stacked).float().unsqueeze(0).to(DEVICE)
        # state = torch.from_numpy(stacked).float().to(DEVICE)
        # state = state.squeeze(0)  # Remove extra dim if present
        torch_image_stack = torch.from_numpy(state['image_stack'] ).unsqueeze(0).float().to(DEVICE)  # torch.from_numpy(stacked).unsqueeze(0).float().to(DEVICE)
        action_map = {
            'STOP': 0,
            'FORWARD': 1,
            'BACKWARD': 2,
            'LEFT': 3,
            'RIGHT': 4
        }
        print(state["prev_action"])

        # state_tensor = {
        #     "image_stack": torch_image_stack,
        #     "servo_angle": torch.tensor([[state["servo_angle"]]], dtype=torch.float32).to(DEVICE),
        #     "prev_action": torch.tensor([[action_map[state["prev_action"]]]], dtype=torch.long).to(DEVICE)
        # }

        # action = correction_heuristic(state['image_stack'], state['servo_angle'], state['prev_action'])
        # if action != -1:
        #     robot.control_action(Action(action))
        #     state = robot.get_state()
        
        # print(f"angle shape: {state_tensor['servo_angle'].shape}")
        # print(f"previous_ action: {state_tensor['prev_action'].shape}")
        
        state_tensor = {
            "image_stack": torch_image_stack,  # shape [1, 4, H, W]
            "servo_angle": torch.tensor([state["servo_angle"]], dtype=torch.float32).unsqueeze(1).to(DEVICE),  # shape [1, 1]
            "prev_action": torch.tensor([action_map[state["prev_action"]]], dtype=torch.long).unsqueeze(1).to(DEVICE)  # shape [1, 1]
        }

        with torch.no_grad():
            pi_e = bc_model.get_action_probs(state_tensor)
        action_idx = pi_e.argmax(dim=1).item()
        # action_idx = agent.select_action(state, mode="mean")
        print("choseen action ", action_idx)
        action = Action(int(action_idx))

        robot.control_action(action)
        time.sleep(robot.calculate_sample_interval())

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    robot.close()
    print("Shutdown complete")
