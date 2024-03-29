from random import random
from torch import nn, Tensor
import torch
import math
import numpy as np
import torch.distributions as dist


def lecun_normal_init(tensor: Tensor, scale: float = 1.0, theshold: float = 1.0) -> None:
    """
    Creates an initializer for truncated normal distribution.

    Args:
        scale: Input tensor.
        scale: scaling factor (positive float).
    Returns:
        tensor: Initialized tensor.
    """
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    variance = math.sqrt(scale / fan_in)
    stddev = variance / .87962566103423978

    # sqrt2 = math.sqrt(2)

    nn.init.uniform_(tensor, -2, 2)
    return tensor * stddev

