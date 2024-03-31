import torch
from pkg.train.module.loss import EuclideanDistanceMSE


def test_euclidean_distance_mse():
    true = torch.tensor([[1, 2, 3], [4, 5, 6]])

    pred = torch.tensor([[2, 2, 3], [3, 5, 7]])

    loss = EuclideanDistanceMSE()
    assert loss(pred, true).item() == 1.2071068286895752


if __name__ == "__main__":
    test_euclidean_distance_mse()
