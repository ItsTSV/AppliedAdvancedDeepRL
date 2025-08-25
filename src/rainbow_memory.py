import numpy as np
import torch
from collections import deque
from tensordict import TensorDict
from torchrl.data import (
    TensorDictReplayBuffer,
    PrioritizedSampler,
    LazyTensorStorage,
    RoundRobinWriter,
)


class PrioritizedExperienceReplay:
    """Wrapper for TorchRL buffer"""

    def __init__(
        self,
        memory_size: int,
        alpha: float,
        beta: float,
        batch_size: int,
        device,
        gamma: float,
        n_step: int,
    ):
        """Inits Prioritized Experience Replay with given parameters"""
        # PER
        self.device = device
        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=memory_size, device=device),
            sampler=PrioritizedSampler(
                memory_size,
                alpha=alpha,
                beta=beta,
            ),
            writer=RoundRobinWriter(),
            priority_key="td_error",
            batch_size=batch_size,
        )

        # N-Step buffer
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_cache = deque(maxlen=self.n_step)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Adds episode experience to PER"""
        # Add to n-step buffer
        n_step_done, data = self._add_to_n_step_cache(
            state, action, reward, next_state, done
        )
        if not n_step_done:
            return

        # If n-step calculation is complete, add to PER
        state, action, reward, next_state, done = data
        data = TensorDict(
            {
                "state": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                "next_state": torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
                "action": torch.tensor(action, dtype=torch.int64).unsqueeze(0),
                "reward": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
                "done": torch.tensor(done, dtype=torch.float32).unsqueeze(0),
                "td_error": torch.tensor(1.0, dtype=torch.float32).unsqueeze(0),
            },
            batch_size=[1],
        ).to(self.device)
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

    def _add_to_n_step_cache(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> tuple[bool, tuple]:
        """Adds transition to n-step cache, processes it.

        Returns:
            bool: Is transitions processes?
            tuple: If yes, contains processed episode info. If not, contains None.
        """
        self.n_step_cache.append((state, action, reward, next_state, done))
        if len(self.n_step_cache) < self.n_step:
            return False, ()

        total_reward = 0
        state, action, _, _, _ = self.n_step_cache[0]
        next_state = self.n_step_cache[-1][3]
        done = self.n_step_cache[-1][4]

        for index, (_, _, reward, _, done) in enumerate(self.n_step_cache):
            total_reward += (self.gamma**index) * reward
            if done:
                next_state = self.n_step_cache[index][3]
                done = self.n_step_cache[index][4]
                break

        return True, (state, action, total_reward, next_state, done)
