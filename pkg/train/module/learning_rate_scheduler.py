from torch.optim.lr_scheduler import LRScheduler


class DefaultLRScheduler(LRScheduler):
    def __init__(self, optimizer, gamma=1.0, last_epoch=-1):
        self.gamma = gamma
        super(DefaultLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
