import torch.nn as nn

RELU = "relu"
TANH = "tanh"


def get_activation(activation: str):
    if activation == RELU:
        return nn.ReLU()
    elif activation == TANH:
        return nn.Tanh()
    else:
        raise ValueError("please type in the correct activation function")
