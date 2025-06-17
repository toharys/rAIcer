import torch.nn as nn
import torch
import torch.nn.functional as F
from q_networks import CNNQnetwork
from torch.utils.data import DataLoader
from rAIcer_env import Action

class BehaviorCloningPolicy(nn.Module):
    """
    Used for estimate the action dist. π_e(a|s), i.e. the behavior policy - the policy that generated action in
    the offline dataset
    """
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.encoder = CNNQnetwork(in_channels, num_actions)  # logits over actions

    def forward(self, x):
        return self.encoder(x)

    def get_action_probs(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)



def train_bc_model(train_set, in_channels, num_actions=5, batch_size=64, num_epochs=10,
                   lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu", model=None):
    """
    Trains a Behavior Cloning model to estimate π_e(a|s)

    :param train_set: Dataset of (stacked_frames, action)
    :param in_channels: Number of input channels (e.g., 1 for grayscale)
    :param num_actions: Number of discrete actions
    :param batch_size: Batch size for training
    :param num_epochs: Number of epochs
    :param lr: Learning rate
    :param device: CUDA or CPU
    :return: Trained BehaviorCloningPolicy model
    """
    bc_model = model if model else BehaviorCloningPolicy(in_channels, num_actions).to(device)
    optimizer = torch.optim.Adam(bc_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    bc_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device).long()
            logits = bc_model(states)
            per_sample_loss = F.cross_entropy(logits, actions, reduction='none')

            weights = torch.ones_like(actions, dtype=torch.float, device=device)
            weights[actions != Action.STOP.value] = 5.0

            loss = (weights * per_sample_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # avg_loss = total_loss / len(train_loader)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return bc_model


if __name__ == '__main__':
    # Sanity Check
    model = BehaviorCloningPolicy(in_channels=1, num_actions=5)     # grayscale, 5 actions
    dummy_input = torch.randn(8, 1, 64, 64)     # [batch_size, channels, height, width]
    logits = model(dummy_input)     # shape: [8, 5]
    probs = model.get_action_probs(dummy_input)     # shape [8, 5], sums to 1 across dim=1 (probs)

    print("Logits:", logits.shape)
    print("Probabilities:", probs.shape)
    print("Prob sum per sample:", probs.sum(dim=1))  # should be close to 1