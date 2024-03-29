import torch
from torch import nn,Tensor


def get_loss_fn(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "euclidean_distance_mse":
        return EuclideanDistanceMSE()
    else:
        raise ValueError(f"loss name is not correct {name}")


class EuclideanDistanceMSE(nn.Module):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        rmse = torch.sqrt((torch.sum((y_true - y_pred) ** 2, dim=-1)))
        return torch.mean(rmse)

