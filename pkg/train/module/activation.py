import torch.nn as nn

# Constants for supported activation functions
RELU = "relu"  # Rectified Linear Unit
TANH = "tanh"  # Hyperbolic Tangent
RReLU = "rrelu"  # Randomized Rectified Linear Unit
LeakyReLU = "LeakyReLU"  # Leaky Rectified Linear Unit


def get_activation(activation: str) -> nn.Module:
    """Get PyTorch activation function module by name.

    Creates and returns the appropriate PyTorch activation function module based on
    the provided activation name.

    Args:
        activation (str): Name of the activation function to use

    Returns:
        nn.Module: PyTorch activation function module

    Raises:
        ValueError: If activation name is not one of the supported options
    """
    if activation == RELU:
        return nn.ReLU()
    elif activation == TANH:
        return nn.Tanh()
    elif activation == RReLU:
        return nn.RReLU()
    elif activation == LeakyReLU:
        return nn.LeakyReLU()
    else:
        raise ValueError("please type in the correct activation function")
