import argparse
import time
from typing import List

import numpy as np

from common.constant import HDF5, TEST_NAME, TRAIN_NAME, VALIDATION_NAME
from pkg.data_utils.dataset_generation import split_dataset_indices
from pkg.train.datasets.utils import import_data_config
from task.passive_biv.data.datasets_preparation_hdf5 import PassiveBiVPreparationDataset

data_format = HDF5

if __name__ == "__main__":
    np.random.seed(753)

    parser = argparse.ArgumentParser(description="Model Selection")

    parser.add_argument("--model_name", type=str, default="fe_heart_sage_v3", help="model name")

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    start_time = time.time()

    model_name = args[0].model_name

    data_config = import_data_config("passive_biv", model_name, "passive_biv")

    data_config["sample_path"] = f"{data_config['task_data_path']}/record_inputs"

    # generate sample indices
    sample_indices_dict = split_dataset_indices(
        data_config["sample_path"],
        data_config["train_split_ratio"],
    )

    sample_indices_dict[TEST_NAME] = sample_indices_dict[VALIDATION_NAME]

    # generate dataset
    for data_type in [TRAIN_NAME, VALIDATION_NAME, TEST_NAME]:
        print(f"start {data_type} {sample_indices_dict[data_type]}")
        data_config["sample_indices"] = sample_indices_dict[data_type]

        if data_format == HDF5:
            data = PassiveBiVPreparationDataset(data_config, data_type)
        else:
            raise ValueError(f"please check the accuracy of data_format {data_format}")

        data.prepare_dataset_process()

    print(f"data preparation done, total time: {time.time() - start_time}s")
