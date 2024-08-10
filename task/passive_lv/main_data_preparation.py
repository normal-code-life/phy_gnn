import threading
import time

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.utils.monitor import monitor_cpu_usage
from task.passive_lv.fe_heart_sage_v1.data.datasets import import_data_config
from task.passive_lv.fe_heart_sage_v1.data.datasets_preparation import FEHeartSageV1PreparationDataset

if __name__ == "__main__":
    monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(100,))
    monitor_thread.start()

    start_time = time.time()

    config = import_data_config("passive_lv", "fe_heart_sage_v1")

    for data_type in [TRAIN_NAME, VALIDATION_NAME]:
        data_preprocess = FEHeartSageV1PreparationDataset(config, data_type)

        data_preprocess.prepare_dataset_process()

    monitor_thread.join()

    print(f"total time: {time.time() - start_time}s")
