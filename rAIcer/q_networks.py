import torch
import torch.nn as nn


class CNNQnetwork(nn.Module):
    """
    Atari-style CNN from DeepMind's DQN paper
    """
    def __init__(self, in_channels: int, num_actions: int):
        """
        A CNN Q-network
        :param in_channels: Number of input channels (3 for RGB)
        :param num_actions: Number of discrete actions in the environment
        """
        super().__init__()

        # 3 convolutional layers, ReLU as activation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # [batch,32,23,23]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # [batch,64,10,10]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # [batch, 64, 8, 8]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # ensures output is always [64, 8, 8]
        )

        # 2 fully-connected layers, using ReLU as activation function
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)  # [batch_size, 64, 8, 8]
        b = a.view(a.size(0), -1)  # [batch_size, 64*8*8]
        c = self.fc(b)  # [batch_size, num_actions]
        return c

class SmallCNNQnetwork(nn.Module):
    """
    Introduce a bottleneck in the Q network
    """
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))    # ensures output is always 8x8
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResidualCNNQnetwork(nn.Module):
    """
    ResNet style CNN, introduce residual connection inside the conv block
    """
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.resblock = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.downsample = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((9, 9))  # ensures fixed output size
        self.relu = nn. ReLU()
        self.fc = nn.Sequential(
            nn.Linear(64*9*9, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.relu(self.initial(x))
        residual = x
        x = self.resblock(x)
        x += residual   # residual connection
        x = self.relu(x)
        x = self.downsample(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DropoutCNNQnetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.AdaptiveAvgPool2d((8, 8))  # Ensures fixed conv output size
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class EnsembeleCNNQnetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.q_networks = nn.ModuleList([
            CNNQnetwork(in_channels, num_actions),
            SmallCNNQnetwork(in_channels, num_actions),
            ResidualCNNQnetwork(in_channels, num_actions),
            DropoutCNNQnetwork(in_channels, num_actions)
        ])

    def forward(self, x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """
        :param x:
        :param mode: one of the next - "mean", "min" or "all"
        :return:
            - "mean": mean Q-values across all networks
            - "min": element-wise min Q-values
            - "all": list of individual Q-values from all networks
        """
        outputs = [q_net(x) for q_net in self.q_networks]  # list of [batch, num_actions] tensors
        if mode == "mean":
            return torch.stack(outputs).mean(dim=0)     # [batch, num_actions]
        elif mode == "min":
            return torch.stack(outputs).min(dim=0).values   # conservative
        elif mode == "all":
            return outputs
        else:
            raise ValueError(f"Unknown ensemble mode: {mode}")


if __name__=="__main__":
    # Sanity Check
    model = EnsembeleCNNQnetwork(in_channels=1, num_actions=5)
    dummy_input = torch.randn(8, 1, 640, 480)  # [batch_size, channels, height, width]

    # Forward pass through
    outputs = model(dummy_input, "mean")

    print(f"Mean Q-values shape: {outputs.shape}")
    print(f"Mean Q-values sample:\n{outputs[0]}")