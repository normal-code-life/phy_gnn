import torch
from torch import Tensor, nn


class EuclideanDistanceMSE(nn.Module):
    """Euclidean Distance Mean Squared Error loss function.

    Computes the mean squared error between predicted and true values using
    Euclidean distance. First calculates the root mean squared error (RMSE)
    along the last dimension, then takes the mean across all other dimensions.

    Args:
        y_pred (Tensor): Predicted values tensor
        y_true (Tensor): Ground truth values tensor

    Returns:
        Tensor: Scalar loss value
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        rmse = torch.sqrt((torch.sum((y_true - y_pred) ** 2, dim=-1)))
        return torch.mean(rmse)
