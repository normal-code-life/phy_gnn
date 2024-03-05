import torch


def get_loss_fn(name: str) -> torch.nn.Module:
    if name == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"loss name is not correct {name}")
