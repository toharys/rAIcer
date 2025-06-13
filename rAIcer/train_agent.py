import os
import pickle
import torch
from agent import train_rAIcer_agent_on_chunk
from tqdm import tqdm
from behavior_cloning_policy import BehaviorCloningPolicy
from train_bc import load_bc_chunk



UPDATE_BC = True

MODELS_DIR = "./models"
BC_MODEL_PATH = "bc_model.pt"
CHUNKS_DIR = "./dataset"
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "trained_agent.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_agent():
    # Sort chunk files by index
    chunk_files = sorted(
        [f for f in os.listdir(CHUNKS_DIR) if f.startswith("replay_buffer_chunk_") and f.endswith(".pkl")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    print(f"Found {len(chunk_files)} chunks")

    bc = None
    if UPDATE_BC:
        bc_path = os.path.join(MODELS_DIR, BC_MODEL_PATH)

        # Load sample to infer in_channels
        sample_file = sorted([f for f in os.listdir(CHUNKS_DIR) if f.startswith("bc_dataset_chunk_")])[0]
        sample_path = os.path.join(CHUNKS_DIR, sample_file)
        # sample_data = torch.load(sample_path, map_location=DEVICE)
        sample_data = load_bc_chunk(sample_path)
        sample_state = sample_data[0][0]
        in_channels = sample_state.shape[0]

        bc = BehaviorCloningPolicy(in_channels=in_channels, num_actions=5)
        bc.load_state_dict(torch.load(bc_path, map_location=DEVICE))
        bc.eval()
        # with open(bc_path, "rb") as f:
        #     bc = BehaviorCloningPolicy(in_channels=in_channels, num_actions=5)
        #     bc.load_state_dict(torch.load(bc_path))
        #     bc.eval()
    start_chunk = 1  # todo: change it if required
    for i, fname in tqdm(enumerate(chunk_files[start_chunk:], start=start_chunk)):
        print(f"Training on chunk {i}")

        data_path = os.path.join(CHUNKS_DIR, fname)
        with open(data_path, "rb") as f:
            buffer = pickle.load(f)

        train_rAIcer_agent_on_chunk(
            chunk_index=i,
            replay_buffer=buffer,
            bc_model=bc,
            agent_checkpoint=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None,
            save_path=CHECKPOINT_PATH,
            batch_size=16,
            update_bc_model=UPDATE_BC
        )

        torch.cuda.empty_cache()
        del buffer
        import gc
        gc.collect()
