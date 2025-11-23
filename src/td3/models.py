import torch
import torch.nn as nn


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

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, network_size),
            nn.ReLU(),
            nn.Linear(network_size, action_space_size),
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
