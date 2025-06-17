import socket
import pickle
import torch
from agent import rAIcerAgent
from robot_control import Action

HOST = '10.100.102.26'  # Jetson's IP
PORT = 4000


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Connected to Jetson")