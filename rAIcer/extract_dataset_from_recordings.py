import os
import pickle
import torch
from rAIcer_env import compute_reward, Action
from data_preprocessing import H5PyDataset, save_bc_to_h5
from replay_buffer import ReplayBuffer
from tqdm import tqdm

DATA_DIR = "./new_samples"
OUTPUT_DIR = "./dataset"

REPLAY_SAVE_PATH = "replay_buffer.pkl"
BC_SAVE_PATH = "bc_dataset.h5"
BUFFER_CAPACITY = 100_000
CHUNK_SIZE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_chunk(files, chunk_index):
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    bc_dataset = []

    for filename in tqdm(files, desc=f"→ Chunk {chunk_index}", leave=False):
        dataset = H5PyDataset(os.path.join(DATA_DIR, filename))
        previous_action = None

        for i in range(len(dataset)):
            sample = dataset[i]
            current_stack = sample["current_state"].numpy()
            next_stack = sample["next_state"].numpy()
            action_str = sample["action"].lower()

            try:
                current_action = Action[action_str.upper()]
            except KeyError:
                print(f"Unknown action: {action_str}, skipping.")
                continue

            reward, done = compute_reward(current_stack, current_action, previous_action)

            state_tensor = torch.from_numpy(current_stack).float()
            replay_buffer.push(
                state_tensor,
                current_action.value,
                reward,
                torch.from_numpy(next_stack).float(),
                done
            )
            bc_dataset.append((state_tensor, current_action.value))
            previous_action = current_action

        dataset.close()

    # Save chunk
    replay_path = os.path.join(OUTPUT_DIR, f"replay_buffer_chunk_{chunk_index}.pkl")
    bc_path = os.path.join(OUTPUT_DIR, f"bc_dataset_chunk_{chunk_index}.h5")
    print(f"Saving chunk {chunk_index}")
    with open(replay_path, "wb") as f:
        pickle.dump(replay_buffer, f)
    save_bc_to_h5(bc_dataset, bc_path)
    print(f"✓ Saved chunk {chunk_index} → {replay_path}, {bc_path}")



def process_all_h5_files(data_dir=DATA_DIR) -> tuple[ReplayBuffer, list]:
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    bc_dataset = []
    h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]

    print("load h5 files")
    for filename in tqdm(h5_files, desc="Processing h5 files"):
        dataset = H5PyDataset(os.path.join(data_dir, filename))

        previous_action = None
        for i in tqdm(range(len(dataset)), desc=f"→ {filename}", leave=False):
            sample = dataset[i]
            current_stack = sample["current_state"].numpy()
            next_stack = sample["next_state"].numpy()
            action_str = sample["action"].lower()
            try:
                current_action = Action[action_str.upper()]
            except KeyError:
                print(f"Unknown action: {action_str}, skipping.")
                continue

            reward, done = compute_reward(current_stack, current_action, previous_action)

            # Convert to tensors for replay buffer
            state_tensor = torch.from_numpy(current_stack).float()
            replay_buffer.push(
                state_tensor,
                current_action.value,
                reward,
                torch.from_numpy(next_stack).float(),
                done
            )
            bc_dataset.append((state_tensor, current_action.value))

            previous_action = current_action

        dataset.close()

    return replay_buffer, bc_dataset


if __name__ == "__main__":
    # h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".h5")])
    # todo ~~~~~~~~~~~~~~~~~~~~~~~~ REMOVE
    h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".h5") and int(f.split("_")[-1].split(".")[0]) > 124])
    print(f"Found {len(h5_files)} files.")

    for filename in h5_files:
        index = int(filename.split("_")[-1].split(".")[0])  # Extract index from 'only_line_99.h5'
        try:
            process_chunk([filename], index)  # Pass single file and use its index for saving
        except OSError as e:
            print(f"Skipping file {filename} due to error: {e}")

    # # Process in chunks
    # start_chunk = 0  # todo
    # total_chunks = (len(h5_files) + CHUNK_SIZE - 1) // CHUNK_SIZE
    # for chunk_num in range(start_chunk, total_chunks):
    #     start_idx = chunk_num * CHUNK_SIZE
    #     chunk_files = h5_files[start_idx:start_idx + CHUNK_SIZE]
    #     process_chunk(chunk_files, chunk_num)




