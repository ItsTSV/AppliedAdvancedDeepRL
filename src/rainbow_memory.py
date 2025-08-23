import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import (
    TensorDictReplayBuffer,
    PrioritizedSampler,
    LazyTensorStorage,
)


class PrioritizedExperienceReplay:
    """Wrapper for TorchRL buffer"""

    def __init__(self, memory_size: int, alpha: float, beta: float, batch_size: int, device):
        """Inits Prioritized Experience Replay with given parameters"""
        self.device = device
        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=memory_size, device=device),
            sampler=PrioritizedSampler(
                memory_size,
                alpha=alpha,
                beta=beta,
            ),
            priority_key="td_error",
            batch_size=batch_size,
        )

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Adds episode experience to PER"""
        data = TensorDict({
            "state": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            "next_state": torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
            "action": torch.tensor(action, dtype=torch.int64).unsqueeze(0),
            "reward": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
            "done": torch.tensor(done, dtype=torch.float32).unsqueeze(0),
            "td_error": torch.tensor(1.0, dtype=torch.float32).unsqueeze(0)
        }, batch_size=[1]).to(self.device)
        self.buffer.add(data)

    def sample(self, batch_size) -> tuple:
        """Sample batch and return states, actions, rewards, next_states, dones, weights, indices"""
        batch, info = self.buffer.sample(batch_size, return_info=True)

        # Squeeze to prevent weird tensor shape mismatch
        states = batch["state"].squeeze(1)
        actions = batch["action"]
        rewards = batch["reward"]
        next_states = batch["next_state"].squeeze(1)
        dones = batch["done"]
        weights = batch["td_error"]
        indices = info["index"]

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors):
        """Update TD errors for given indices"""
        self.buffer.update_priority(indices, td_errors)
