import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import csv
from typing import Union, TypedDict


class StateDict(TypedDict):
    image_stack: torch.Tensor
    servo_angle: torch.Tensor   # assuming batched float tensor
    prev_action: torch. Tensor  # assuming batched int tensor

class BehaviorCloningPolicy(nn.Module):
    """
    Used for estimate the action dist. π_e(a|s), i.e. the behavior policy - the policy that generated action in
    the offline dataset
    """
    def __init__(self, in_channels, num_actions):
        super().__init__()

        # Pre encoder to emphasize subtle motion cues from stacked frames
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        # Encode using ResNet, modify input and output
        self.encoder = models.resnet18(weights=None)
        # self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()  # Remove classification head
        self.encoder.avgpool = nn.Identity()

        # Feature projector (down to 256 dims)
        self.projector = nn.Conv2d(512, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Positional embedding (flattened spatial locations)
        self.pos_embed = nn.Parameter(torch.randn(16, 1, 256)*0.02)  # 4x4 = 16 positions

        # Transformer - "focouses on the relevent areas in the frame"
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(0.3)

        # MLP embedding for the angle and action
        self.extrs_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # MLP decoder
        self.mlp = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions)
        )

    def forward(self, state: StateDict = None, stacked_frames=None, mode="full_state"):
        """
        state = { 'image_stack': Tensor (4, H, W),
                    'servo_angle': float,
                    'prev_action': int
                    }
        mode: "full_state" or "stack_only"
        """
        device = next(self.parameters()).device

        if mode == "full_state":
            image_stack = state['image_stack'].to(device)  # shape: [B, 4, H, W]
            servo_angle = state['servo_angle'].to(device).view(-1, 1).float()
            prev_action = state['prev_action'].to(device).view(-1, 1).float()
        elif mode == "stack_only":
            image_stack = stacked_frames.to(device)
            B = image_stack.size(0)
            servo_angle = torch.full((B, 1), -1.0, device=device)
            prev_action = torch.full((B, 1), -1.0, device=device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Encoder + projection
        x = self.pre_encoder(image_stack)   # [B, 3, H, W]
        x = self.encoder.conv1(x)     # [B, 512, H', W'] via AdaptiveAvgPool
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.projector(x)   # [B, 256, H', W']
        x = self.pool(x)    # [B, 256, 4, 4]

        # Pos embeddings
        B, C, H, W = x.shape
        seq = x.view(B, C, -1).permute(2, 0, 1)     # [S=16, B, 256]
        pos_embed = self.pos_embed[:H * W]  # [S=16, 1, 256]
        seq = seq + pos_embed   # Broadcast over batch

        # Transformer
        transformed = self.transformer(seq)     # [S, B, 256]
        features = self.dropout(transformed.mean(dim=0))  # [B, 256]

        # Concatenate with angle and prev action embeddings
        extras = torch.cat([servo_angle, prev_action], dim=1)   # [B, 2]
        extras_embed = self.extrs_mlp(extras)   # [B, 64]
        final_input = torch.cat([features, extras_embed], dim=1)# [B, 320]

        # Decoder
        logits = self.mlp(final_input)  # shape: [B, num_actions]
        return logits

    def get_action_probs(self, state: StateDict, mode="full_state"):
        logits = self.forward(state, mode)
        return F.softmax(logits, dim=-1)

    def get_action(self, state, mode="full_state"):
        probs = self.get_action_probs(state)
        return torch.argmax(probs, dim=1)  # returns tensor of shape [B]


def gradient_step(states: StateDict, actions, device, bc_model, optimizer, mode, mirror):
    MIRROR_ACTION = torch.tensor([0, 1, 2, 4, 3], device=device)

    if mode == "full_state":
        image_stack = states["image_stack"].to(device)
        servo_angle = states["servo_angle"].to(device)
        prev_action = states["prev_action"].to(device)
        actions = actions.to(device)

        if mirror:
            image_stack = torch.flip(image_stack, dims=(3, ))   # Horizontal flip
            servo_angle = 160.0 - servo_angle    # Mirror around center
            prev_action = MIRROR_ACTION[prev_action.long()]
            actions = MIRROR_ACTION[actions.long()]

        states = {
            "image_stack": image_stack,
            "servo_angle": servo_angle,
            "prev_action": prev_action
        }

    elif mode == "stack_only":
        states = states.to(device)
        actions = actions.to(device)

        if mirror:
            states = torch.flip(states, dims=(3, ))     # Horizontal flip
            actions = MIRROR_ACTION[actions.long()]

    else:
        raise Exception(f"INVALID MODE: {mode},  Expected full_state or stack_only ")

    logits = bc_model(states, mode)
    per_sample_loss = F.cross_entropy(logits, actions, reduction='none')
    weights = torch.ones_like(actions, dtype=torch.float, device=device)
    loss = (weights * per_sample_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_bc_model(chunk_idx, train_set, in_channels, num_actions=5, batch_size=64, num_epochs=10,
                   lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu", model=None, mode="full_state"):
    """
    Trains a Behavior Cloning model to estimate π_e(a|s)

    :param train_set: Dataset of (state, action)
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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    bc_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for states, actions in train_loader:
            states, actions = states, actions.to(device).long()
            # Original record
            total_loss += gradient_step(states, actions, device, bc_model, optimizer, mode, mirror=False)

            # Mirror record
            total_loss += gradient_step(states, actions, device, bc_model, optimizer, mode, mirror=True)

        avg_loss = total_loss / (2*len(train_loader))  #(2*len(train_loader))
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        log_loss_per_epoch(chunk_idx=chunk_idx, epoch_num=epoch, avg_loss=avg_loss)
        # log_avg_losses(chunk_idx=chunk_idx, epoch_num=epoch,  loss_log={"total_loss": [avg_loss]})

    log_avg_loss_per_chunk(chunk_idx=chunk_idx, avg_loss=avg_loss)
    # log_avg_losses(chunk_idx=chunk_idx, loss_log={"total_loss": [avg_loss]})
    return bc_model

def log_loss_per_epoch(chunk_idx, epoch_num, avg_loss, log_file="loss_per_epoch_bc.csv"):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["chunk_index", "epoch", "avg_loss"])
        writer.writerow([chunk_idx, epoch_num, avg_loss])

def log_avg_loss_per_chunk(chunk_idx, avg_loss, log_file="avg_loss_per_chunk_bc.csv"):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["chunk_index", "avg_total_loss"])
        writer.writerow([chunk_idx, avg_loss])


# def log_avg_losses(chunk_idx=None, epoch_num=None, loss_log=None, log_file="avg_losses_per_chunk_bc.csv"):
#
#     avg_total_loss = sum(loss_log["total_loss"]) / len(loss_log["total_loss"])
#     file_exists = os.path.isfile(log_file)
#
#     with open(log_file, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(["chunk_index", "avg_total_loss"])
#         writer.writerow([epoch_num, avg_total_loss])


if __name__ == '__main__':
    import torch

    # Create dummy state input
    B, C, H, W = 4, 4, 120, 160  # batch size, stack size, height, width
    stacked_frames = torch.rand(B, C, H, W)

    # Test FULL_STATE mode
    print("\n[Testing FULL_STATE mode]")
    state_full = {
        "image_stack": stacked_frames.clone(),
        "servo_angle": torch.tensor([80.0, 100.0, 120.0, 140.0]).unsqueeze(1),  # Vary angle
        "prev_action": torch.tensor([0, 1, 2, 3])
    }

    model = BehaviorCloningPolicy(in_channels=4, num_actions=5)
    logits_full = model(state_full, mode="full_state")
    probs_full = torch.softmax(logits_full, dim=1)

    print("[Full] Logits:\n", logits_full)
    print("[Full] Action probabilities:\n", probs_full)
    print("[Full] Predicted actions:", torch.argmax(probs_full, dim=1).tolist())

    # Change only the servo angle to see if it affects predictions
    state_full["servo_angle"] += 20.0
    logits_angle_changed = model(state_full, mode="full_state")
    probs_changed = torch.softmax(logits_angle_changed, dim=1)

    print("[Full+Angle Shift] Action probabilities:\n", probs_changed)
    print("[Full+Angle Shift] Predicted actions:", torch.argmax(probs_changed, dim=1).tolist())

    # Test STACK_ONLY mode
    print("\n[Testing STACK_ONLY mode]")
    logits_stack = model(stacked_frames=stacked_frames.clone(), mode="stack_only")
    probs_stack = torch.softmax(logits_stack, dim=1)

    print("[StackOnly] Logits:\n", logits_stack)
    print("[StackOnly] Action probabilities:\n", probs_stack)
    print("[StackOnly] Predicted actions:", torch.argmax(probs_stack, dim=1).tolist())
