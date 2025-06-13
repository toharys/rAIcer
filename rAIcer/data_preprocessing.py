import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from rAIcer_env import compute_reward, Action
from replay_buffer import ReplayBuffer
# from robot_control import Action
import pickle

class ProcessedReplayDataset(Dataset):
    def __init__(self, transitions):
        """
        transitions: list of tuples (state, action, reward, next_state, done)
        """
        self.data = transitions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.data[idx]
        return {
            'state': torch.from_numpy(state).float(),
            'action': torch.tensor(action, dtype=torch.long),
            'reward': torch.tensor(reward, dtype=torch.float32),
            'next_state': torch.from_numpy(next_state).float(),
            'done': torch.tensor(done, dtype=torch.bool)
        }


class H5PyDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (string): Path to the HDF5 file
        """
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')

        # Get dataset lengths
        self.length = len(self.file['stacks'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get data from HDF5 file
        current_stack = self.file['stacks'][idx]
        action = self.file['actions'][idx]
        action_str = action.decode('utf-8')
        next_stack = self.file['next_frames'][idx]

        # Convert to PyTorch tensors
        current_stack = torch.from_numpy(current_stack).float()
        next_stack = torch.from_numpy(next_stack).float()

        # Convert action to numerical value (you may want to customize this)
        action_map = {
            'stop': 0,
            'forward': 1,
            'backward': 2,
            'left': 3,
            'right': 4
        }
        action_idx = action_map.get(action.decode('utf-8'), 0)  # Default to 'stop' if unknown

        return {
            'current_state': current_stack,
            'action': action_str,
            'next_state': next_stack
        }

    def close(self):
        self.file.close()


def load_h5py_data(file_path, batch_size=32, shuffle=True):
    # Create dataset
    dataset = H5PyDataset(file_path)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Adjust based on your system
    )
    return dataloader, dataset

def save_to_h5(processed_data, save_path):
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('states', data=np.array([d[0] for d in processed_data]))
        f.create_dataset('actions', data=np.array([d[1] for d in processed_data]))
        f.create_dataset('rewards', data=np.array([d[2] for d in processed_data]))
        f.create_dataset('next_states', data=np.array([d[3] for d in processed_data]))
        f.create_dataset('dones', data=np.array([d[4] for d in processed_data]))

def save_bc_to_h5(bc_data, save_path):
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('states', data=np.array([d[0] for d in bc_data]), compression='lzf')
        f.create_dataset('actions', data=np.array([d[1] for d in bc_data]), compression='lzf')


# prev_state, action, current_state
def data_preprocessing(dataset_path: str) -> tuple[ReplayBuffer, list]:
    """
    assert that it getting a path to dataset where each sample is a list of dict 'current_state', 'action', n
    """
    # Load the data
    replay_buffer = ReplayBuffer(100_000)   # consist of (state, action, reward, next_state, done)
    bc_dataset = []   # consist of (state, action)

    dataloader, dataset = load_h5py_data(dataset_path, batch_size=1, shuffle=False)
    previous_action = None

    for batch in dataloader:
        action = batch['action'][0]
        current_state: np.ndarray = batch['current_state'].squeeze(0).numpy()
        next_state: np.ndarray = batch['next_state'].squeeze(0).numpy()

        # action_str = batch['action'][0].decode('utf-8')  # from torch tensor to str
        action_enum = Action[action.upper()]
        prev_action_enum = Action(previous_action) if previous_action is not None else None

        reward, done = compute_reward(current_state, action_enum, prev_action_enum)

        state_tensor = torch.from_numpy(current_state).float()
        next_state_tensor = torch.from_numpy(next_state).float()

        replay_buffer.push(state_tensor, action_enum.value, reward, next_state_tensor, done)
        bc_dataset.append((state_tensor, action_enum.value))

        previous_action = action_enum

    dataset.close()
    with open("replay_buffer.pkl", "wb") as f:
        pickle.dump(replay_buffer, f)

    save_bc_to_h5(bc_dataset, "bc_dataset.h5")

    return replay_buffer, bc_dataset


if __name__ == "__main__":
    # dataloader, dataset = load_h5py_data("binary_frame_stacks_with_commands.h5")
    # print(print(list(dataset.file.keys())))
    # data_preprocessing("binary_frame_stacks_with_commands.h5")
    # Load the saved replay buffer
    with open("replay_buffer.pkl", "rb") as f:
        replay_buffer = pickle.load(f)

    # Sample one transition
    sample = replay_buffer.sample(1) # get first (and only) sample

    # Print the tuple length
    print("Length of tuple:", len(sample))
