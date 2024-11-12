import argparse
import threading
import time
from typing import List

import numpy as np

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.train.datasets.base_datasets import import_data_config
from pkg.utils.monitor import monitor_cpu_usage
from task.passive_lv.data.datasets_preparation import FEHeartSagePreparationDataset

np.random.seed(753)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Selection")

    parser.add_argument("--model_name", type=str, default="", help="model name")

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(10,))
    monitor_thread.start()

    start_time = time.time()

    model_name = args[0].model_name

    config = import_data_config("passive_lv", model_name, "lvData")

    for data_type in [TRAIN_NAME, VALIDATION_NAME]:
        data_preprocess = FEHeartSagePreparationDataset(config, data_type)

        data_preprocess.prepare_dataset_process()

    monitor_thread.join()

    print(f"data preparation done, total time: {time.time() - start_time}s")
