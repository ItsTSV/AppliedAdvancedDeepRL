import torch
import torch.nn as nn
import numpy as np
from src.shared.weight_initializer import init_layer


class QNet(nn.Module):
    """Q-Network for predicting values; serves as a critic in TD3."""

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int, init_method: str):
        super().__init__()
        self.network = nn.Sequential(
            init_layer(nn.Linear(state_space_size + action_space_size, network_size),
                       method=init_method, gain=np.sqrt(2)),
            nn.ReLU(),
            init_layer(nn.Linear(network_size, network_size), method=init_method, gain=np.sqrt(2)),
            nn.ReLU(),
            init_layer(nn.Linear(network_size, 1), method=init_method, gain=1.0),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Returns:
            torch.Tensor: Estimated Q-value of action in a state
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ActorNet(nn.Module):
    """Policy network for TD3."""

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int, init_method: str):
        super().__init__()
        self.network = nn.Sequential(
            init_layer(nn.Linear(state_space_size, network_size), method=init_method, gain=np.sqrt(2)),
            nn.ReLU(),
            init_layer(nn.Linear(network_size, network_size), method=init_method, gain=np.sqrt(2)),
            nn.ReLU(),
            init_layer(nn.Linear(network_size, action_space_size), method=init_method, gain=0.01),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forwards pass through the network.

         Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Actions to advance the environment
        """
        return self.network(x)
