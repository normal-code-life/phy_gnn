from torch.utils.data import DataLoader

from task.passive_biv.data.datasets import import_data_config
from task.passive_biv.data.datasets_train import PassiveBiVTrainDataset

if __name__ == "__main__":
    data_config = import_data_config()

    train_data = PassiveBiVTrainDataset(data_config, "train")

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=data_config.get("batch_size", 1),
        num_workers=2,
        prefetch_factor=1,
    )

    s = 0
    for i in range(4):
        for inputs, labels in train_data_loader:
            s += 1
            for i in inputs:
                print(i, inputs[i].shape)
            for i in labels:
                print(i, labels[i].shape)
