import torch
import torch.nn as nn
import numpy as np


def init_layer(layer: nn.Module, method: str = "default", gain: float = np.sqrt(2), bias: float = 0.0):
    """Initializes layer weights

    Args:
        method (string): "orthogonal" for ortho init, whatever else for default PyTorch init.
    """
    if method == "orthogonal" and isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias)

    return layer
