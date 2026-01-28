import torch
import torch.nn as nn
import numpy as np
from src.shared.weight_initializer import init_layer


class QNet(nn.Module):
    """Q-Network for predicting values; serves as a critic in SAC."""

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space_size + action_space_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Returns:
            torch.Tensor: Estimated Q-value of action in a state
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ActorNet(nn.Module):
    """Policy network for SAC."""

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, network_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(network_size, action_space_size)
        self.log_std_head = nn.Linear(network_size, action_space_size)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forwards pass through the network.

         Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            tuple: Means of gaussian distribution, log standard deviation
        """
        x = self.network(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        return mean, log_std
