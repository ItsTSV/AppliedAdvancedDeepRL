from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class RolloutStep:
    """Data structure to hold a single step in the rollout buffer."""

    state: np.ndarray
    action: np.ndarray
    logprob: float
    reward: float
    value: float
    finished: bool

    def unpack(self):
        """Unpacks the step data into a tuple."""
        return (
            self.state,
            self.action,
            self.logprob,
            self.reward,
            self.value,
            self.finished,
        )


class RolloutBuffer:
    """Buffer that serves as a storage for PPO episode data."""

    def __init__(self):
        """Initialize a list that holds PPO episode info"""
        self.memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        state: np.array,
        action: np.array,
        logprob: float,
        reward: float,
        value: float,
        finished: bool,
    ):
        """Adds a new step to the buffer."""
        self.memory.append(RolloutStep(state, action, logprob, reward, value, finished))

    def clear(self):
        """Clears the buffer, removing all stored data"""
        del self.memory[:]

    def get_tensors(self) -> tuple:
        """Returns memory contents as PyTorch tensors"""
        states, actions, logprobs, rewards, values, dones = zip(
            *[step.unpack() for step in self.memory]
        )
        return (
            torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(logprobs, dtype=torch.float32, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(values, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
        )
