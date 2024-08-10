import random

import torch
from torch.utils.data import DataLoader

from pkg.train.datasets.shuffle_iterable_datasets import ShuffledIterableDataset

if __name__ == "__main__":
    data = list(range(50))

    buffer_size = 10

    # 创建数据集
    dataset = ShuffledIterableDataset(data, buffer_size)

    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=4)

    # 迭代数据加载器
    batch_num = 0
    for batch in data_loader:
        batch_num += 1
        print(batch_num, batch)
