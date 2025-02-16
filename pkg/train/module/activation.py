import torch.nn as nn

RELU = "relu"
TANH = "tanh"
RReLU = "rrelu"
LeakyReLU = "LeakyReLU"


def get_activation(activation: str):
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
