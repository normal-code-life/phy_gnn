import torch.nn as nn

RELU = "relu"
TANH = "tanh"
RReLU = "rrelu"


def get_activation(activation: str):
    if activation == RELU:
        return nn.ReLU()
    elif activation == TANH:
        return nn.Tanh()
    elif activation == RReLU:
        return nn.RReLU
    else:
        raise ValueError("please type in the correct activation function")
