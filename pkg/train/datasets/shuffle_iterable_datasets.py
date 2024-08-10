import random

from torch.utils.data import IterableDataset

from pkg.train.datasets.base_datasets_train import BaseIterableDataset


class ShuffledIterableDataset(BaseIterableDataset):
    def __init__(self, dataset, buffer_size):
        super(IterableDataset, self).__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def buffer_shuffle(self, data_iter):
        buffer = []
        try:
            # Fill the buffer initially
            for _ in range(self.buffer_size):
                buffer.append(next(data_iter))
        except StopIteration:
            pass

        while buffer:
            # Shuffle the buffer and yield one item
            random.shuffle(buffer)
            yield buffer.pop()

            try:
                # Refill the buffer if more data is available
                buffer.append(next(data_iter))
            except StopIteration:
                pass

    def __iter__(self):
        data_iter = iter(self.dataset)
        return iter(self.buffer_shuffle(data_iter))
