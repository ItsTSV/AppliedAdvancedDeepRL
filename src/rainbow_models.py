import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Implementation of NoisyNets paper: https://arxiv.org/abs/1706.10295v3"""

    def __init__(self, input_size: int, output_size: int, sigma_init: float = 0.5):
        """Creates Noisy layer with given input and output size

        input_size (int): Dimensions of input tensor
        output_size (int): Dimensions of output tensor
        sigma_init (float): How much noise is added to weight and bias
        """
        super().__init__()
        # IO
        self.input_size = input_size
        self.output_size = output_size
        self.sigma_init = sigma_init

        # Sigma
        self.sigma_weight = nn.Parameter(torch.empty(output_size, input_size))
        self.sigma_bias = nn.Parameter(torch.empty(output_size))

        # Mu
        self.mu_weight = nn.Parameter(torch.empty(output_size, input_size))
        self.mu_bias = nn.Parameter(torch.empty(output_size))

        # Epsilon
        self.register_buffer("epsilon_weight", torch.empty(output_size, input_size))
        self.register_buffer("epsilon_bias", torch.empty(output_size))

        # Init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Resets sigma and mu values."""
        # Mu
        mu_range = 1 / math.sqrt(self.input_size)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)

        # Sigma
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.input_size))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.output_size))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Applies scaling to the generated noise.

        Generate random vector from Normal Distribution(0, 1) and apply transformation from the paper
        f(x) = sign(x) * sqrt(|x|). This reduces number of generated features from input_size × output_size to
        input_size + output_size."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Resets noise values in networks.

        Generates row and column noise vectors, creates nosie matrix by computing outer product;
        sets epsilon weight and bias accordingly.
        """
        column_noise_vector = self._scale_noise(self.input_size)
        row_noise_vector = self._scale_noise(self.output_size)

        # Generate whole noise matrix (M_ij = row[i] * column[j])
        self.epsilon_weight.copy_(row_noise_vector.ger(column_noise_vector))

        # Bias is only row noise
        self.epsilon_bias.copy_(row_noise_vector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass via layer."""
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        return F.linear(x, weight, bias)


class DuelingQRDQN(nn.Module):
    """Dueling Deep Q-Network for Rainbow Agent"""

    def __init__(
        self,
        action_space_size: int,
        observation_space_size: int,
        use_noisy: int,
        quantile_count: int,
    ):
        """Initialize network with given action and observation size"""
        super().__init__()

        # Switch for noisy vs. standard nets
        linear = NoisyLinear if use_noisy else nn.Linear

        # QR-DQN
        self.quantile_count = quantile_count
        self.action_space_size = action_space_size

        # Shared fully connected part
        self.network = nn.Sequential(
            linear(observation_space_size, 128),
            nn.ReLU(),
            linear(128, 128),
            nn.ReLU(),
        )

        # Value and advantage heads
        self.advantage_head = linear(128, action_space_size * self.quantile_count)
        self.value_head = linear(128, self.quantile_count)

    def reset_noise(self):
        """Resets NoisyNets"""
        self.advantage_head.reset_noise()
        self.value_head.reset_noise()
        for layer in self.network:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Returns:
            tensor: [Batch, Actions, Quantiles]
        """
        x = self.network(x)

        # Dueling
        values = self.value_head(x)
        advantages = self.advantage_head(x)

        # QR-DQN
        advantages = advantages.view(-1, self.action_space_size, self.quantile_count)
        q_values = values.unsqueeze(1) + (
            advantages - advantages.mean(dim=1, keepdim=True)
        )
        return q_values


class DuelingConvDQN(nn.Module):
    """Convolutional Dueling Deep Q-Network for Rainbow Agent"""

    def __init__(
        self,
        action_space_size: int,
        observation_space_size: int,
        use_noisy: int,
        quantile_count: int,
    ):
        """Initializes network with given action and observation size"""
        super().__init__()

        # Switch for noisy vs. standard nets
        linear = NoisyLinear if use_noisy else nn.Linear

        # QR-DQN
        self.quantile_count = quantile_count
        self.action_space_size = action_space_size

        # Shared convolutional part
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
            linear(3136, 512), nn.ReLU(), linear(512, action_space_size)
        )
        self.value_head = nn.Sequential(linear(3136, 512), nn.ReLU(), linear(512, 1))

    def reset_noise(self):
        """Resets NoisyNets"""
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Returns:
            tensor: [Batch, Actions, Quantiles]
        """
        x = self.network(x)
        x = x.view(x.size(0), -1)

        # Dueling
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        # QR-DQN
        advantages = advantages.view(-1, self.action_space_size, self.quantile_count)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
