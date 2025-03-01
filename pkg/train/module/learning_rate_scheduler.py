from torch.optim.lr_scheduler import LRScheduler


class DefaultLRScheduler(LRScheduler):
    """Default learning rate scheduler that applies a multiplicative factor.

    Multiplies the learning rate of each parameter group by a fixed gamma value
    at each step. This provides a simple way to decay the learning rate by a
    constant factor.

    Args:
        optimizer: Wrapped optimizer
        gamma (float): Multiplicative factor of learning rate decay. Default: 1.0
        last_epoch (int): The index of last epoch. Default: -1

    Example:
        >>> # Assuming optimizer uses lr = 0.1 for all groups
        >>> # lr = 0.1     if epoch = 0
        >>> # lr = 0.1*0.9 if epoch = 1
        >>> # lr = 0.1*0.9*0.9 if epoch = 2
        >>> scheduler = DefaultLRScheduler(optimizer, gamma=0.9)
    """

    def __init__(self, optimizer, gamma=1.0, last_epoch=-1):
        self.gamma = gamma
        super(DefaultLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rates using current gamma value.

        Returns:
            list: Updated learning rates for each parameter group
        """
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
