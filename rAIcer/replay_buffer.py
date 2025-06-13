import torch
import random
# from rAIcer_env import Action
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self,
             state: torch.Tensor,
             action: int,
             reward: float,
             next_state: torch.Tensor,
             done: bool):
        """
        Stores a transition in the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random batch of transitions.
        Returns:
          Tuple of (states, actions, rewards, next_states, dones) as NumPy arrays.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),  # shape: [B, C, H, W]
            torch.tensor(actions, dtype=torch.long),  # shape: [B]
            torch.tensor(rewards, dtype=torch.float32),  # shape: [B]
            torch.stack(next_states),  # shape: [B, C, H, W]
            torch.tensor(dones, dtype=torch.float32)  # shape: [B]
        )

    def __len__(self) -> int:
        return len(self.buffer)