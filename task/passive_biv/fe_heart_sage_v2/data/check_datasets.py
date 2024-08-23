import time

from torch.utils.data import DataLoader

from task.passive_biv.fe_heart_sage_v2.data.datasets import import_data_config
from task.passive_biv.fe_heart_sage_v2.data.datasets_train_hdf5 import FEHeartSageV2TrainDataset

if __name__ == "__main__":
    data_config = import_data_config()

    train_data = FEHeartSageV2TrainDataset(data_config, "train")

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=data_config.get("batch_size", 1),
        num_workers=1,
        prefetch_factor=None,
    )

    start_time = time.time()

    s = 0
    for i in range(4):
        for inputs, labels in train_data_loader:
            s += 1
            # for i in inputs:
            #     print(i, inputs[i].shape)
            # for i in labels:
            #     print(i, labels[i].shape)

            print(inputs["index"].item())

        print(f"{i}: {time.time() - start_time}s")
