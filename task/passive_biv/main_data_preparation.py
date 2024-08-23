import time

import numpy as np

from common.constant import HDF5, TRAIN_NAME, VALIDATION_NAME, TFrecord
from pkg.data.utils.dataset_generation import split_dataset_indices
from task.passive_biv.fe_heart_sage_v2.data import datasets_preparation_hdf5, datasets_preparation_tfrecord
from task.passive_biv.fe_heart_sage_v2.data.datasets import import_data_config

data_format = HDF5

if __name__ == "__main__":
    np.random.seed(753)

    start_time = time.time()

    data_config = import_data_config()

    # generate sample indices
    sample_indices_dict = split_dataset_indices(
        data_config["sample_path"],
        data_config["train_split_ratio"],
    )

    # generate dataset
    for data_type in [TRAIN_NAME, VALIDATION_NAME]:
        data_config["sample_indices"] = sample_indices_dict[data_type]

        if data_format == TFrecord:
            data = datasets_preparation_tfrecord.PassiveBiVPreparationDataset(data_config, data_type)
        elif data_format == HDF5:
            data = datasets_preparation_hdf5.PassiveBiVPreparationDataset(data_config, data_type)
        else:
            raise ValueError(f"please check the accuracy of data_format {data_format}")

        data.prepare_dataset_process()

    print(f"data preparation done, total time: {time.time() - start_time}s")
