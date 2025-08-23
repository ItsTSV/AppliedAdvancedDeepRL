import torch
import torch.nn as nn


"""
    This is only temporary solution, as Rainbow DQN uses Noisy Nets.
    Soon, this will be replaced with a proper implementation.
"""


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network for Rainbow Agent"""

    def __init__(self, action_space_size: int, observation_space_size: int):
        """Initialize network with given action and observation size"""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_space_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.advantage_head = nn.Linear(128, action_space_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass through the network.

        Args:
            x (torch.tensor): Input tensor representing the state.

        Returns:
            torch.tensor: Q-values for each action.

        """
        x = self.network(x)
        values = self.value_head(x)
        advantages = self.advantage_head(x)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DuelingConvDQN(nn.Module):
    """Convolutional Dueling Deep Q-Network for Rainbow Agent"""

    def __init__(self, action_space_size: int, observation_space_size: int):
        """Initializes network with given action and observation size"""
        super().__init__()
        # Convolutional part
        self.network = nn.Sequential(
            nn.Conv2d(observation_space_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Value and Advantage heads
        self.advantage_head = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, action_space_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass through the network.

        Args:
            x (torch.tensor): Input tensor representing the state.

        Returns:
            torch.tensor: Q-values for each action.
        """
        x = self.network(x)
        x = x.view(x.size(0), -1)
        advantages = self.advantage_head(x)
        values = self.value_head(x)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
