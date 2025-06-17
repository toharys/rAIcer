import h5py
import torch
import plotly.graph_objects as go
import pandas as pd
from enum import Enum
import pickle
import os
from agent import rAIcerAgent
from rAIcer_env import Action
from collections import Counter
from tqdm import tqdm

# Constants
RECORD_DIR ="./samples"
AGENT_PATH = "./models/trained_agent.pt"
CHUNKS_DIR = "./dataset"
LOSS_LOGGING_PATH = "avg_losses_per_chunk.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAP_ACT = {b'stop':0, b'left':1, b'right':2, b'forward':3, b'backward':4}

def test_rAIcer():
    # Load trained agent
    try:
        checkpoint = torch.load(AGENT_PATH, map_location=DEVICE)
        in_channels = 4
        num_actions = 5
        agent = rAIcerAgent(state_shape=(in_channels, 120, 160), num_actions=num_actions)
        agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        agent.q_network.to(DEVICE)
        agent.q_network.eval()
    except:
        print("error: there is no trained agent")

    with h5py.File(RECORD_DIR+"/record_1.h5", "r") as f:
        states = f["stacks"][:]     # shape (N, H, W)
        actions = f["actions"][:]   # shape (N, )

    results = []
    for i in range(len(states)):
        state_tensor = torch.from_numpy(states[i])
        with torch.no_grad():
            action_idx = agent.select_action(state_tensor, mode="all")
        predicted = Action(int(action_idx))
        expert = actions[i]
        print(f"    agent action: {predicted}")
        print(f"    expert action: {expert}")
        results.append((predicted, expert))

    correct = sum(p.value == MAP_ACT[e] for p, e in results)
    accuracy = correct / len(results)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Accuracy: {accuracy:.2%}")


def action_frequencies():
    h5_files = sorted([f for f in os.listdir(RECORD_DIR) if f.endswith(".h5")])
    total_actions = 0
    expert_counts = Counter()

    for fname in tqdm(h5_files):
        try:
            with h5py.File(os.path.join(RECORD_DIR, fname), "r") as f:
                actions = f["actions"][:]
                total_actions += len(actions)

                for expert_action in actions:
                    expert_counts[expert_action] += 1
        except OSError as e:
            print(f"Skipping invalid file: {fname} ({e})")


    print("\nExpert action frequencies:")
    for action, count in expert_counts.items():
        print(f"  {action} taken {count} times ({100 * count / total_actions:.2f}%)")

    # making histogram
    labels = [str(action_name) for action_name in expert_counts.keys()]
    counts = list(expert_counts.values())

    fig = go.Figure(data=[
        go.Bar(x=labels, y=counts, marker_color='lightskyblue')
    ])

    fig.update_layout(
        title="Expert Action Frequency Histogram",
        xaxis_title="Action",
        yaxis_title="Count",
        template="plotly_white"
    )

    # Save as HTML (interactive) and PNG (static)
    fig.write_html("expert_action_histogram.html")
    fig.write_image("expert_action_histogram.png")

    print("✓ Saved interactive HTML and static PNG histograms")


def average_reward():
    chunk_files = sorted(
        [f for f in os.listdir(CHUNKS_DIR) if f.startswith("replay_buffer_chunk_") and f.endswith(".pkl")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    total_rewaed = 0.0
    total_transitions = 0.0

    for fname in tqdm(chunk_files, desc="Calculating average reward"):
        path = os.path.join(CHUNKS_DIR, fname)
        try:
            with open(path, "rb") as f:
                buffer = pickle.load(f)

            for transition in buffer.buffer:
                _, _, reward, _, _ = transition
                total_rewaed += reward
                total_transitions += 1

        except Exception as e:
            print(f"SKipping {fname} due to error: {e}")

    if total_transitions == 0:
        print("No transitions found.")
        return

    avg_reward = total_rewaed / total_transitions
    print(f"\n✓ Average reward across all chunks: {avg_reward:.4f}")


def plot_avg_losses():
    # Load CSV
    df = pd.read_csv(LOSS_LOGGING_PATH)

    # Create a line plot for each loss component
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["chunk_index"],
        y=df["avg_total_loss"],
        mode="lines+markers",
        name="Avg Total Loss"
    ))
    fig.add_trace(go.Scatter(
        x=df["chunk_index"],
        y=df["avg_bellman_loss"],
        mode="lines+markers",
        name="Avg Bellman Loss"
    ))
    fig.add_trace(go.Scatter(
        x=df["chunk_index"],
        y=df["avg_kl_div"],
        mode="lines+markers",
        name="Avg KL Divergence"
    ))

    # Update layout
    fig.update_layout(
        title="Average Losses per Chunk",
        xaxis_title="Chunk Index",
        yaxis_title="Loss Value",
        template="plotly_white"
    )

    # Save to HTML
    output_html = "avg_losses_plot.html"
    fig.write_html(output_html)
    print(f"✓ Plot saved to {output_html}")

if __name__ == "__main__":
    test_rAIcer()
    # plot_avg_losses()