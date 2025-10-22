import torch
import torch.nn as nn


class QNet(nn.Module):
    """Q-Network for predicting values; serves as a critic in SAC."""

    def __init__(self, action_space_size: int, state_space_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space_size + action_space_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        """Forward pass through the network.

        Returns:
            torch.tensor: Estimated Q-value of action in a state
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ActorNet(nn.Module):
    """Policy network for SAC."""

    def __init__(self, action_space_size: int, state_space_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_space_size)
        self.log_std_head = nn.Linear(256, action_space_size)

    def forward(self, x: torch.tensor) -> tuple:
        """Forwards pass through the network.

         Args:
            x (torch.tensor): Input tensor representing the state.

        Returns:
            tuple: Means of gaussian distribution, log standard deviation
        """
        x = self.network(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        return mean, log_std
