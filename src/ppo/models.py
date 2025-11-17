import torch
import torch.nn as nn


class DiscreteActorCriticNet(nn.Module):
    """Discrete Actor-Critic Network for PPO.

    Uses a shared network for both actor and critic heads. This will come in handy
    after this network is applied to environments with continuous action spaces,
    where convolutional layers will be needed.
    """

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_space_size, network_size),
            nn.Tanh(),
            nn.Linear(network_size, network_size),
            nn.Tanh(),
            nn.Linear(network_size, network_size),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(network_size, action_space_size)
        self.critic_head = nn.Linear(network_size, 1)

    def forward(self, x: torch.tensor) -> tuple:
        """Forward pass through the network.

        Args:
            x (torch.tensor): Input tensor representing the state.

        Returns:
            tuple: Actor output (action probabilities) and critic output (state value).
        """
        x = self.network(x)
        return self.actor_head(x), self.critic_head(x)


class ContinuousActorCriticNet(nn.Module):
    """Continuous Actor-Critic Network for PPO.

    Uses separate networks for actor and critic. This increases the number of
    parameters, but allows agent to train more efficiently.
    """

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_space_size, network_size),
            nn.Tanh(),
            nn.Linear(network_size, network_size),
            nn.Tanh(),
            nn.Linear(network_size, network_size),
            nn.Tanh(),
            nn.Linear(network_size, action_space_size),
        )

        self.actor_log_st = nn.Parameter(torch.zeros(action_space_size))

        self.critic = nn.Sequential(
            nn.Linear(state_space_size, network_size),
            nn.Tanh(),
            nn.Linear(network_size, network_size),
            nn.Tanh(),
            nn.Linear(network_size, network_size),
            nn.Tanh(),
            nn.Linear(network_size, 1),
        )

    def forward(self, x: torch.tensor) -> tuple:
        """Forward pass through the network.

        Args:
            x (torch.tensor): Input tensor representing the state.

        Returns:
            tuple: Means of gaussian distribution, log standard deviation, value function output.
        """
        mean = self.actor(x)
        log_std = self.actor_log_st.expand_as(mean)
        value = self.critic(x)
        return mean, log_std, value
