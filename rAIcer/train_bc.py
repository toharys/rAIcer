import os
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from behavior_cloning_policy import train_bc_model, BehaviorCloningPolicy
from tqdm import tqdm

CHUNKS_DIR = "./dataset"
BC_MODEL_PATH = "./models/bc_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS_PER_CHUNK = 2

def load_bc_chunk(h5_path):
    with h5py.File(h5_path, 'r') as f:
        states = torch.from_numpy(f['states'][:]).float()
        actions = torch.from_numpy(f['actions'][:]).long()
        return TensorDataset(states, actions)

def train_bc():
    chunk_files = sorted([
        f for f in os.listdir(CHUNKS_DIR)
        if f.startswith("bc_dataset_chunk_") and f.endswith(".h5")
    ])

    model = None
    sample_path = os.path.join(CHUNKS_DIR, chunk_files[0])
    sample_data = load_bc_chunk(sample_path)
    in_channels = sample_data[0][0].shape[0]
    num_actions = 5

    # Load model checkpoint if it exists
    if os.path.exists(BC_MODEL_PATH):
        checkpoint = torch.load(BC_MODEL_PATH, map_location=DEVICE)

        model = BehaviorCloningPolicy(in_channels, num_actions).to(DEVICE)
        model.load_state_dict(checkpoint)

        print(f"✓ Loaded existing model from {BC_MODEL_PATH}")

    for i, fname in tqdm(enumerate(chunk_files)):
        chunk_path = os.path.join(CHUNKS_DIR, fname)
        try:
            dataset = load_bc_chunk(chunk_path)

            # Initialize the model only once
            if model is None:
                # Get in_channels from sample shape
                sample_state = dataset[0][0]
                in_channels = sample_state.shape[0]
                model = BehaviorCloningPolicy(in_channels, num_actions)

            # # Load weights if checkpoint exists
            # if os.path.exists(BC_MODEL_PATH):
            #     model.load_state_dict(torch.load(BC_MODEL_PATH))

            model = train_bc_model(
                train_set=dataset,
                in_channels=in_channels,
                num_actions=num_actions,
                batch_size=BATCH_SIZE,
                num_epochs=EPOCHS_PER_CHUNK,
                lr=1e-4,
                device=DEVICE,
                model=model     # continue training on the same model
            )

            # Save after each chunk
            torch.save(
                model.state_dict(),
                BC_MODEL_PATH
            )
            print(f"✓ Checkpoint saved to {BC_MODEL_PATH}")
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ Skipping chunk {fname} due to error: {e}")
            continue

