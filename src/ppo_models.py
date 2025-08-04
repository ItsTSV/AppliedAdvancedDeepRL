import torch
import torch.nn as nn


class ActorCriticNet(nn.Module):
    """Discrete Actor-Critic Network for PPO."""
    def __init__(self, action_space_size: int, state_space_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(32, action_space_size)
        self.critic_head = nn.Linear(32, 1)

    def forward(self, x: torch.tensor):
        """Forward pass through the network.

        Args:
            x (torch.tensor): Input tensor representing the state.

        Returns:
            tuple: Actor output (action probabilities) and critic output (state value).
        """
        x = self.network(x)
        return self.actor_head(x), self.critic_head(x)
