import torch
import torch.nn as nn
import numpy as np


def init_layer(layer: nn.Module, method: str = "default", gain: float = np.sqrt(2), bias: float = 0.0):
    """Initializes layer weights

    Args:
        method (string): "orthogonal" for ortho init, whatever else for default PyTorch init.
    """
    if isinstance(layer, nn.Linear):
        if method == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain)
            torch.nn.init.constant_(layer.bias, bias)
        elif method == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    return layer
