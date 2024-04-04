import torch
from torch import Tensor, nn


class EuclideanDistanceMSE(nn.Module):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        rmse = torch.sqrt((torch.sum((y_true - y_pred) ** 2, dim=-1)))
        return torch.mean(rmse)
