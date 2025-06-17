import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from q_networks import EnsembeleCNNQnetwork
from behavior_cloning_policy import BehaviorCloningPolicy, train_bc_model
from rAIcer_env import Action
from replay_buffer import ReplayBuffer

SAVE_DIR = "./models/trained_agent.py"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_UPDATES = 10_000

class rAIcerAgent:
    def __init__(self,
                 state_shape: tuple[int, ...],
                 #replay_buffer: ReplayBuffer,   # contains the training samples
                 num_actions: int = 5,
                 lr: float = 1e-4,
                 batch_size: int = 16,
                 gamma: float = 0.99,
                 #buffer_capacity: int = 100_000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        # self.robot = Robot()
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.lr = lr

        self.step_counter = 0

        # Q-Network and target Q-Network
        in_channels: int = state_shape[0]
        self.q_network: nn.Module = EnsembeleCNNQnetwork(in_channels, num_actions).to(device)
        self.target_q_network: nn.Module = EnsembeleCNNQnetwork(in_channels, num_actions).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # Initialize the target Q-net as the Q-net
        self.update_freq: int = 1000    # target network update frequency

        # Behavior Cloning network
        self.bc_net: nn.Module = BehaviorCloningPolicy(in_channels, num_actions)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience Replay
        # self.replay_buffer: ReplayBuffer = ReplayBuffer(buffer_capacity)
        #self.replay_buffer: ReplayBuffer = replay_buffer

    def select_action(self, state: np.ndarray, mode: str = "mean") -> int:
        """
        Select an action using the current Q-network (no exploration - offline RL approach)
        :param state: stack of 4 binary frames, shape [4, H, W] (as np.ndarray or torch.Tensor)
        :param mode: "mean" or "min" - how tp combine ensemble Qs
        :return: int - chosen action index (the one that maximize the expected return)
        """
        # Convert the numpy array to tensor
        if isinstance(state, np.ndarray):
            tensor = torch.from_numpy(state).float() / 255.0  # Normalize to [0, 1], shape [4, H, W]
        else:
            tensor = state.float() / 255.0  # Already tensor

        state = tensor.unsqueeze(0)  # Add batch dimension: [1, 4, H, W]

        self.q_network.eval()
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.q_network(state, mode=mode)
            action = q_values.argmax(dim=1).item()
        return action

    def update(self, replay_buffer, lambda_kl: float = 0.1, temperature: float = 1.0) -> dict[str, float]:
        """
        Samples a batch from the replay buffer and does one gradient update step on the Q-network
        :param replay_buffer
        :param lambda_kl: the coefficient of the KL-div term in the loss
        :param temperature:
        :return: logging of the losses
        """
        if len(replay_buffer) < self.batch_size:
            return None  # Not enough data

        # Sample a batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # (1) Get Q(s,a) (the trained policy) for action taken
        self.q_network.train()
        self.target_q_network.eval()
        q_values = self.q_network(states, mode="mean")  # shape: [B, num_actions]
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # shape: [B]

        # (2) Compute Q-target from target network
        with torch.no_grad():
            q_next = self.target_q_network(next_states, mode="mean")  # shape: [B, num_actions]
            q_next_max = q_next.max(dim=1)[0]  # best Q-value for next state
            q_target = rewards + self.gamma * q_next_max * (1 - dones)

        # (3) Calculate the loss
        # bellman_loss = F.mse_loss(q_sa, q_target)   # Bellman loss
        bellman_error = (q_sa - q_target).pow(2)
        weights = torch.ones_like(bellman_error)
        weights[actions != Action.STOP.value] = 5.0

        bellman_loss = (weights * bellman_error).mean()

        # KL divergence to behavior policy
        with torch.no_grad():
            pi_e = self.bc_net.get_action_probs(states)     # shape: [B, num_actions]
        pi = F.softmax(q_values / temperature, dim=-1)
        kl_div = F.kl_div(pi.log(), pi_e, reduction='batchmean')

        total_loss = bellman_loss + lambda_kl*kl_div

        # (4) Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # (5) Periodically update target network
        self.step_counter += 1
        if self.step_counter % self.update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            "total_loss": total_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "kl_div": kl_div.item()
        }


def plot_losses_per_chunk(chunk_index, loss_log):
    # Plot the 3 losses
    plt.figure(figsize=(10, 6))
    plt.plot(loss_log["steps"], loss_log["total_loss"], label="Total Loss")
    plt.plot(loss_log["steps"], loss_log["bellman_loss"], label="Bellman Loss")
    plt.plot(loss_log["steps"], loss_log["kl_div"], label="KL Divergence")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves for Chunk {chunk_index}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./loss_plot_chunk_{chunk_index}.png")
    plt.close()

def log_avg_losses(chunk_index, loss_log, log_file="avg_losses_per_chunk.csv"):
    avg_total_loss = sum(loss_log["total_loss"]) / len(loss_log["total_loss"])
    avg_bellman_loss = sum(loss_log["bellman_loss"]) / len(loss_log["bellman_loss"])
    avg_kl_div = sum(loss_log["kl_div"]) / len(loss_log["kl_div"])

    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["chunk_index", "avg_total_loss", "avg_bellman_loss", "avg_kl_div"])
        writer.writerow([chunk_index, avg_total_loss, avg_bellman_loss, avg_kl_div])


def train_rAIcer_agent_on_chunk(chunk_index, replay_buffer, bc_model, agent_checkpoint, save_path, batch_size, update_bc_model):

    sample_state = replay_buffer.buffer[0][0]
    state_shape = sample_state.shape

    if agent_checkpoint and os.path.exists(agent_checkpoint):
        print(f"[Chunk {chunk_index}] Loading agent from checkpoint...")
        agent = rAIcerAgent(state_shape=state_shape, batch_size=batch_size)
        checkpoint = torch.load(agent_checkpoint, map_location=DEVICE)
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.step_counter = checkpoint.get('step_counter', 0)
        if 'bc_model_state_dict' in checkpoint:
            agent.bc_net.load_state_dict(checkpoint['bc_model_state_dict'])
            agent.bc_net.to(DEVICE)

    else:
        print(f"[Chunk {chunk_index}] Initializing new agent...")
        agent = rAIcerAgent(state_shape=state_shape, batch_size=batch_size)


    # Load pre-trained BC model
    if update_bc_model:
        agent.bc_net.load_state_dict(bc_model.state_dict())
    agent.bc_net.to(DEVICE)
    agent.bc_net.eval()

    agent.q_network.to(DEVICE)
    agent.q_network.train()

    print(f"[Chunk {chunk_index}] Training agent...")

    loss_log = {
        "steps": [],
        "total_loss": [],
        "bellman_loss": [],
        "kl_div": []
    }
    num_updates = (len(replay_buffer) // batch_size)*5  # "5 epochs"
    for step in tqdm(range(1, num_updates + 1), desc=f"Chunk {chunk_index} Training", unit="step"):
        losses = agent.update(replay_buffer)

        if losses:
            loss_log["steps"].append(step)
            loss_log["total_loss"].append(losses["total_loss"])
            loss_log["bellman_loss"].append(losses["bellman_loss"])
            loss_log["kl_div"].append(losses["kl_div"])

    # plot_losses_per_chunk(chunk_index, loss_log)
    log_avg_losses(chunk_index, loss_log)

    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_q_network_state_dict': agent.target_q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'bc_model_state_dict': agent.bc_net.state_dict(),
        'step_counter': agent.step_counter,
    }, save_path)
    print(f"[Chunk {chunk_index}] Agent saved to {save_path}")





def train_rAIcer_agent(dataset, num_updates=NUM_UPDATES,
                       batch_size=64,
                       update_log_interval=500,
                       save_path="trained_agent.pt"):
    """
    Trains the rAIcerAgent in a pure offline RL setup using a fixed dataset.

    :param dataset: list of transitions (state, action, reward, next_state, done)
    :param num_updates: number of gradient steps to perform
    :param batch_size: minibatch size for each update
    :param update_log_interval: how often to print/log the loss
    :param save_path: path to save the trained agent
    :return: Trained rAIcerAgent
    """
    # (1) Train the behavior cloning model for KL-div regularization
    bc_model = train_bc_model(dataset)

    # (2) Initialize agent and assign BC model
    agent = rAIcerAgent(batch_size=batch_size)
    agent.bc_net = bc_model
    agent.store_static_transitions(dataset)

    # (3) Offline training loop
    for step in tqdm(range(1, num_updates + 1), desc="Offline RL Training", unit="step"):
        losses = agent.update()

        if step % update_log_interval == 0 and losses:
            print(f"[Step {step}] Total: {losses['total_loss']:.4f} | "
                  f"Bellman: {losses['bellman_loss']:.4f} | "
                  f"KL: {losses['kl_div']:.4f}")

    # (4) Save the trained agent (Q-network, optimizer state, etc.)
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_q_network_state_dict': agent.target_q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'bc_model_state_dict': agent.bc_net.state_dict() if agent.bc_net else None,
        'step_counter': agent.step_counter,
    }, save_path)

    print(f"Trained agent saved to {save_path}")

    return agent

# Dummy offline dataset: 100 transitions of random tensors
def create_dummy_dataset(num_samples=100, in_channels=4, height=64, width=64, num_actions=5):
    dataset = []
    for _ in range(num_samples):
        state = torch.randn(in_channels, height, width)
        action = torch.randint(0, num_actions, (1,)).item()
        reward = torch.randn(1).item()
        next_state = torch.randn(in_channels, height, width)
        done = torch.rand(1).item() > 0.9  # 10% done
        dataset.append((state, action, reward, next_state, done))
    return dataset


if __name__ == "__main__":
    dummy_dataset = create_dummy_dataset()
    agent = train_rAIcer_agent(dummy_dataset, num_updates=100, batch_size=8, update_log_interval=20)
    checkpoint = torch.load("trained_agent.pt")
    # an example to agent loading
    checkpoint = torch.load("trained_agent.pt")
    agent = rAIcerAgent()
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.bc_net.load_state_dict(checkpoint['bc_model_state_dict'])
    agent.step_counter = checkpoint.get('step_counter', 0)
