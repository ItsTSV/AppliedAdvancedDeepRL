import numpy as np
import torch


class ReplayBuffer:
    """Buffer that serves as a storage for SAC & TD3 experience data."""

    def __init__(self, capacity: int, action_dim: int, state_dim: int):
        """Initializes a replay buffer for storing experiences."""
        self.capacity = capacity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Adds a new experience to the buffer, recalculates the positions."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """Samples a batch of random experiences from the buffer."""
        indexes = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.states[indexes], device=self.device)
        actions = torch.tensor(self.actions[indexes], device=self.device)
        rewards = torch.tensor(self.rewards[indexes], device=self.device)
        next_states = torch.tensor(self.next_states[indexes], device=self.device)
        dones = torch.tensor(self.dones[indexes], device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return self.size
