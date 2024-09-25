from torch.utils.data import DataLoader

from task.passive_lv.fe_heart_sage_v1.data.datasets import import_data_config
from task.passive_lv.fe_heart_sage_v1.data.datasets_train import FEHeartSageV1TrainDataset

if __name__ == "__main__":
    data_config = import_data_config("passive_lv", "fe_heart_sage_v1")

    train_data = FEHeartSageV1TrainDataset(data_config, "train")

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
            for x in inputs:
                print(x, inputs[x].shape)

            print("labels", labels.shape)
