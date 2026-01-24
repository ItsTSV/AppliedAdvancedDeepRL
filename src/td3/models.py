import torch
import torch.nn as nn
import numpy as np


class QNet(nn.Module):
    """Q-Network for predicting values; serves as a critic in TD3."""

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
    """Policy network for TD3."""

    def __init__(self, action_space_size: int, state_space_size: int, action_low: float,
                 action_high: float, network_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, action_space_size),
            nn.Tanh()
        )

        if np.any(np.isinf(action_low)) or np.any(np.isinf(action_high)):
            print("The bounds are infinite! Assuming the scaling to be handled by the environment.")
            scale = 1.0
            bias = 0.0
        else:
            scale = (action_high - action_low) / 2.0
            bias = (action_high + action_low) / 2.0

        self.register_buffer("action_scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple:
        """Forwards pass through the network.

         Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Actions to advance the environment
        """
        return self.network(x) * self.action_scale + self.action_bias
