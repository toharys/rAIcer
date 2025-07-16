# import io
# import json
# import torch
# import torchvision.models as models
# import torch.nn as nn
# import PIL
# from PIL import Image
# import requests
# import cv2
# from robot_control_ssh_on_laptop_angle_model import Robot, Action
# import time
# import base64

# # Gemini API configuration
# API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
# API_KEY = "AIzaSyAb70zktqGCKzumiL1cl2Z_t8l3VIQHiqU"
# HEADERS = {"Authorization": f"Bearer {API_KEY}"}
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --- Pretrained 12-channel encoder (4×RGB) ---
# class ResNetStackEncoder(nn.Module):
#     def __init__(self, out_dim=512):
#         super().__init__()
#         resnet = models.resnet18(weights="IMAGENET1K_V1")
#         resnet.conv1 = nn.Conv2d(12, 64, 7, 2, 3, bias=False)
#         resnet.fc = nn.Identity()
#         self.backbone = resnet
#         self.projector = nn.Sequential(
#             nn.Linear(512, out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim)
#         )
#     def forward(self, x):
#         with torch.no_grad():
#             feat = self.backbone(x)
#         return self.projector(feat)

# encoder = ResNetStackEncoder()
# encoder.eval()
# print("Encoder initialized successfully")

# def preprocess_and_embed(frames):
#     # frames: numpy array [4,H,W,3] (from FrameStack)
#     tensor = torch.stack([
#         torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255
#         for f in frames  # frames is already a list of 4 numpy arrays
#     ], dim=0)  # [4,3,H,W]
#     input_tensor = tensor.view(1,12,*tensor.shape[2:])  # Combine channels
#     with torch.no_grad():
#         embedding = encoder(input_tensor).squeeze().tolist()
#     return [round(v,4) for v in embedding]

# def encode_frames_to_jpeg(frames):
#     # combine vertically into one tall image (optional)
#     import numpy as np
#     combined = np.vstack(frames)  # [4*H, W, 3]
#     pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
#     buf = io.BytesIO()
#     pil.save(buf, format="JPEG", quality=75)
#     return buf.getvalue()

# def call_gemini(jpeg_bytes, angle, prev_action):
#     print("Connecting to Gemini API...")
    
#     image_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
    
#     payload = {
#         "contents": [{
#             "parts": [
#                 {"text": (
#                     "You are a robot driving assistant that MUST keep moving. NEVER respond with 0 (STOP).\n"
#                     "Valid actions are ONLY:\n"
#                     "0: STOP\n"
#                     "1: FORWARD (always prefer this if possible)\n"
#                     "2: BACKWARD\n"
#                     "3: LEFT\n"
#                     "4: RIGHT\n\n"
#                     f"Current steering angle: {angle}\n"
#                     f"Previous action: {prev_action}\n\n"
#                     "Your mission is to keep the robot moving while centering the white line. "
#                     "Respond ONLY with a single digit (0-4)."
#                     "choose your action base on the stack frame you are gatting, it is 4 frame in a verticall"
#                     "drive cardully and calculated"
#                     "after each action you are taking do 5 stop to see the resolts of your action "
#                     "if you see only black go backward untill you see the white line again"
#                 )},
#                 {
#                     "inline_data": {
#                         "mime_type": "image/jpeg",
#                         "data": image_base64
#                     }
#                 }
#             ]
#         }]
#     }
    
#     params = {'key': API_KEY}
    
#     try:
#         resp = requests.post(API_URL, json=payload, params=params)
#         resp.raise_for_status()
#         print("Successfully connected to Gemini API")
#         response_text = resp.json()['candidates'][0]['content']['parts'][0]['text']
#         return response_text.strip()
#     except Exception as e:
#         print(f"Error connecting to Gemini API: {e}")
#         return "0"  # Default to STOP on error

# def get_next_action(frames, angle, prev_action):
#     jpeg = encode_frames_to_jpeg(frames)
#     action_str = call_gemini(jpeg, angle, prev_action)
#     try:
#         return int(action_str)
#     except ValueError:
#         print(f"Warning: Gemini returned non-integer action: '{action_str}'. Defaulting to STOP (0).")
#         return 0

# # --- Example usage --- #
# if __name__ == "__main__":
#     robot = Robot('10.100.102.26' , 3200)
#     action_map = {
#             'STOP': 0,
#             'FORWARD': 1,
#             'BACKWARD': 2,
#             'LEFT': 3,
#             'RIGHT': 4
#         }
#     try:
#         while not robot.terminate:
#             state = robot.get_state()
#             action = get_next_action(state['image_stack'], state["servo_angle"], action_map[state["prev_action"]])
#             print("Predicted action:", action)
#             robot.control_action(Action(int(action)))
#             time.sleep(robot.calculate_sample_interval())
#     except KeyboardInterrupt:
#         print("Interrupted by user")
#     finally:
#         robot.close()
#         print("Shutdown complete")

# # import io
# # import json
# # import torch
# # import torchvision.models as models
# # import torch.nn as nn
# # from PIL import Image # Corrected import
# # import requests
# # import cv2
# # from robot_control_ssh_on_laptop_angle_model import Robot, Action
# # import time
# # import base64

# # # Import the Google Generative AI library (you are still using requests directly,
# # # but it's good to have this configured if you switch later)
# # import google.generativeai as genai

# # # Gemini API configuration
# # API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
# # API_KEY = "AIzaSyAb70zktqGCKzumiL1cl2Z_t8l3VIQHiqU"
# # # HEADERS = {"Authorization": f"Bearer {API_KEY}"} # Not used with params={'key': API_KEY}
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # # Configure the API key using the client library (even if you're using requests directly for calls)
# # genai.configure(api_key=API_KEY)


# # # --- Pretrained 12-channel encoder (4×RGB) ---
# # class ResNetStackEncoder(nn.Module):
# #     def __init__(self, out_dim=512):
# #         super().__init__()
# #         resnet = models.resnet18(weights="IMAGENET1K_V1")
# #         resnet.conv1 = nn.Conv2d(12, 64, 7, 2, 3, bias=False)
# #         resnet.fc = nn.Identity()
# #         self.backbone = resnet
# #         self.projector = nn.Sequential(
# #             nn.Linear(512, out_dim),
# #             nn.ReLU(),
# #             nn.Linear(out_dim, out_dim)
# #         )
# #     def forward(self, x):
# #         with torch.no_grad():
# #             feat = self.backbone(x)
# #         return self.projector(feat)

# # encoder = ResNetStackEncoder()
# # encoder.eval()
# # print("Encoder initialized successfully")

# # def preprocess_and_embed(frames):
# #     # frames: numpy array [4,H,W,3] (from FrameStack)
# #     tensor = torch.stack([
# #         torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255
# #         for f in frames
# #     ], dim=0)
# #     input_tensor = tensor.view(1,12,*tensor.shape[2:])
# #     with torch.no_grad():
# #         embedding = encoder(input_tensor).squeeze().tolist()
# #     return [round(v,4) for v in embedding]

# # def encode_frames_to_jpeg(frames):
# #     import numpy as np
# #     combined = np.vstack(frames)
# #     pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
# #     buf = io.BytesIO()
# #     pil.save(buf, format="JPEG", quality=75)
# #     return buf.getvalue()

# # def call_gemini(jpeg_bytes, angle, prev_action, max_retries=5):
# #     print("Connecting to Gemini API...")

# #     image_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')

# #     payload = {
# #         "contents": [{
# #             "parts": [
# #                 {"text": (
# #                     "You are a robot driving assistant that MUST keep moving. NEVER respond with 0 (STOP).\n"
# #                     "Valid actions are ONLY:\n"
# #                     "0: STOP\n" # Added back STOP based on your prompt, but keep "MUST keep moving" in mind
# #                     "1: FORWARD (always prefer this if possible)\n"
# #                     "2: BACKWARD\n"
# #                     "3: LEFT\n"
# #                     "4: RIGHT\n\n"
# #                     f"Current steering angle: {angle}\n"
# #                     f"Previous action: {prev_action}\n\n"
# #                     "Your mission is to keep the robot moving while centering the white line. "
# #                     "Respond ONLY with a single digit (0-4)."
# #                     "choose your action base on the stack frame you are gatting, it is 4 frame in a verticall"
# #                     "after each action you are taking do 5 stop to see  the resolts of your action " # This instruction might be tricky for the LLM to follow perfectly.
# #                     "when you see turn turn left or right and drive forward to continue in the road"
# #                     "if you see only black go backward untill you see the white line again"
# #                 )},
# #                 {
# #                     "inline_data": {
# #                         "mime_type": "image/jpeg",
# #                         "data": image_base64
# #                     }
# #                 }
# #             ]
# #         }]
# #     }

# #     params = {'key': API_KEY}
# #     print(f"DEBUG: API_URL being used: {API_URL}")
# #     print(f"DEBUG: Params being sent: {params}")

# #     for attempt in range(max_retries):
# #         try:
# #             resp = requests.post(API_URL, json=payload, params=params)
# #             resp.raise_for_status() # This will raise an HTTPError for 4xx/5xx responses
# #             print("Successfully connected to Gemini API")
# #             response_text = resp.json()['candidates'][0]['content']['parts'][0]['text']
# #             return response_text.strip()
# #         except requests.exceptions.HTTPError as e:
# #             if e.response.status_code == 429:
# #                 wait_time = 2 ** attempt # Exponential backoff: 1, 2, 4, 8, 16 seconds
# #                 print(f"Error connecting to Gemini API (Attempt {attempt+1}/{max_retries}): {e}")
# #                 print(f"Rate limit hit (429). Retrying in {wait_time} seconds...")
# #                 time.sleep(wait_time)
# #             else:
# #                 print(f"Error connecting to Gemini API: {e}")
# #                 return "0" # Other HTTP errors
# #         except Exception as e:
# #             print(f"Error connecting to Gemini API: {e}")
# #             return "0" # General exceptions

# #     print(f"Failed to connect to Gemini API after {max_retries} attempts due to rate limits.")
# #     return "0" # All retries failed

# # def get_next_action(frames, angle, prev_action):
# #     jpeg = encode_frames_to_jpeg(frames)
# #     action_str = call_gemini(jpeg, angle, prev_action)
# #     try:
# #         return int(action_str)
# #     except ValueError:
# #         print(f"Warning: Gemini returned non-integer action: '{action_str}'. Defaulting to STOP (0).")
# #         return 0

# # # --- Example usage --- #
# # if __name__ == "__main__":
# #     robot = Robot('10.100.102.26' , 3200)
# #     action_map = {
# #             'STOP': 0,
# #             'FORWARD': 1,
# #             'BACKWARD': 2,
# #             'LEFT': 3,
# #             'RIGHT': 4
# #         }
# #     try:
# #         while not robot.terminate:
# #             state = robot.get_state()
# #             action = get_next_action(state['image_stack'], state["servo_angle"], action_map[state["prev_action"]])
# #             print("Predicted action:", action)
# #             robot.control_action(Action(int(action)))
# #             time.sleep(robot.calculate_sample_interval()) # This sleep is in addition to any backoff
# #     except KeyboardInterrupt:
# #         print("Interrupted by user")
# #     finally:
# #         robot.close()
# #         print("Shutdown complete")


import io
import json
import torch
import torchvision.models as models
import torch.nn as nn
import PIL
from PIL import Image
import requests
import cv2
from robot_control_ssh_on_laptop_angle_model import Robot, Action
import time
import base64
import threading

# Gemini API configuration
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
API_KEY = "AIzaSyBejSt-ftMz_5_FoBs9YQm10F1_2tkcrgI" #"AIzaSyAb70zktqGCKzumiL1cl2Z_t8l3VIQHiqU"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Pretrained 12-channel encoder (4×RGB) ---
class ResNetStackEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        resnet.conv1 = nn.Conv2d(12, 64, 7, 2, 3, bias=False)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.projector = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim))
        
    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.projector(feat)

encoder = ResNetStackEncoder()
encoder.eval()
print("Encoder initialized successfully")

def preprocess_and_embed(frames):
    # frames: numpy array [4,H,W,3] (from FrameStack)
    tensor = torch.stack([
        torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255
        for f in frames  # frames is already a list of 4 numpy arrays
    ], dim=0)  # [4,3,H,W]
    input_tensor = tensor.view(1,12,*tensor.shape[2:])  # Combine channels
    with torch.no_grad():
        embedding = encoder(input_tensor).squeeze().tolist()
    return [round(v,4) for v in embedding]

def encode_frames_to_jpeg(frames):
    # combine vertically into one tall image (optional)
    import numpy as np
    combined = np.vstack(frames)  # [4*H, W, 3]
    pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=75)
    return buf.getvalue()

class GeminiRequestThread(threading.Thread):
    def __init__(self, jpeg_bytes, angle, prev_action, callback):
        threading.Thread.__init__(self)
        self.jpeg_bytes = jpeg_bytes
        self.angle = angle
        self.prev_action = prev_action
        self.callback = callback
        self.result = None
        
    def run(self):
        self.result = call_gemini(self.jpeg_bytes, self.angle, self.prev_action)
        self.callback(self.result)

def call_gemini(jpeg_bytes, angle, prev_action):
    print("Connecting to Gemini API...")
    
    image_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
    
    payload = {
        "contents": [{
            "parts": [
                {"text": (
                    "You are a robot driving assistant that MUST keep moving. NEVER respond with 0 (STOP).\n"
                    "Valid actions are ONLY:\n"
                    "0: STOP\n"
                    "1: FORWARD (always prefer this if possible)\n"
                    "2: BACKWARD\n"
                    "3: LEFT\n"
                    "4: RIGHT\n\n"
                    f"Current steering angle: {angle} - range 40-120 degresss\n"
                    f"Previous action: {prev_action}\n\n"
                    "Your mission is to keep the robot moving while centering the white line. "
                    "Respond ONLY with a single digit (0-4)."
                    "choose your action base on the stack frame you are gatting, it is 4 frame in a verticall"
                    "drive cardully and calculated"
                    "after each action you are taking do 5 stop to see the resolts of your action "
                    "if you see only black go backward untill you see the white line again"
                    "try to drive mainly in the center of the white line"
                    "use the angle you are geting to change the wills left and right according to the image"
                )},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64
                    }
                }
            ]
        }]
    }
    
    params = {'key': API_KEY}
    
    try:
        resp = requests.post(API_URL, json=payload, params=params)
        resp.raise_for_status()
        print("Successfully connected to Gemini API")
        response_text = resp.json()['candidates'][0]['content']['parts'][0]['text']
        return response_text.strip()
    except Exception as e:
        print(f"Error connecting to Gemini API: {e}")
        return "0"  # Default to STOP on error

def get_next_action(frames, angle, prev_action, robot):
    jpeg = encode_frames_to_jpeg(frames)
    
    # Start with STOP action while waiting for Gemini
    robot.control_action(Action(0))  # Send STOP command
    
    # Create event to track when we get the response
    response_received = threading.Event()
    response_result = [None]  # Using list to allow modification in callback
    
    def callback(result):
        response_result[0] = result
        response_received.set()
    
    # Start the Gemini request thread
    gemini_thread = GeminiRequestThread(jpeg, angle, prev_action, callback)
    gemini_thread.start()
    
    # Wait for the response with a timeout (optional)
    response_received.wait(timeout=10)  # 10 second timeout
    
    if response_result[0] is None:
        print("Warning: Gemini response timed out. Continuing with STOP action.")
        return 0
    
    try:
        return int(response_result[0])
    except ValueError:
        print(f"Warning: Gemini returned non-integer action: '{response_result[0]}'. Defaulting to STOP (0).")
        return 0

# --- Example usage --- #
if __name__ == "__main__":
    robot = Robot('10.100.102.26', 3200)
    action_map = {
            'STOP': 0,
            'FORWARD': 1,
            'BACKWARD': 2,
            'LEFT': 3,
            'RIGHT': 4
        }
    try:
        while not robot.terminate:
            state = robot.get_state()
            action = get_next_action(state['image_stack'], state["servo_angle"], 
                                   action_map[state["prev_action"]], robot)
            print("Predicted action:", action)
            robot.control_action(Action(int(action)))
            time.sleep(robot.calculate_sample_interval())
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        robot.close()
        print("Shutdown complete")