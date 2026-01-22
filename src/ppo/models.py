import torch
import torch.nn as nn
from src.shared.weight_initializer import init_layer


class DiscreteActorCriticNet(nn.Module):
    """Discrete Actor-Critic Network for PPO.

    Uses a shared network for both actor and critic heads. This will come in handy
    after this network is applied to environments with continuous action spaces,
    where convolutional layers will be needed.
    """

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int, init_method: str):
        super().__init__()
        self.network = nn.Sequential(
            init_layer(nn.Linear(state_space_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
            init_layer(nn.Linear(network_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
            init_layer(nn.Linear(network_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
        )
        self.actor_head = init_layer(nn.Linear(network_size, action_space_size), method=init_method, gain=0.01)
        self.critic_head = init_layer(nn.Linear(network_size, 1), method=init_method, gain=1.0)

    def forward(self, x: torch.Tensor) -> tuple:
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

    def __init__(self, action_space_size: int, state_space_size: int, network_size: int, init_method: str):
        super().__init__()
        self.actor = nn.Sequential(
            init_layer(nn.Linear(state_space_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
            init_layer(nn.Linear(network_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
            init_layer(nn.Linear(network_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
            init_layer(nn.Linear(network_size, action_space_size), method=init_method, gain=0.01),
        )

        self.actor_log_st = nn.Parameter(torch.zeros(action_space_size))

        self.critic = nn.Sequential(
            init_layer(nn.Linear(state_space_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
            init_layer(nn.Linear(network_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
            init_layer(nn.Linear(network_size, network_size), method=init_method, gain=5/3),
            nn.Tanh(),
            init_layer(nn.Linear(network_size, 1), method=init_method, gain=1.0),
        )

    def forward(self, x: torch.Tensor) -> tuple:
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
