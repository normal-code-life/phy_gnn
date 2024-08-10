import threading
import time
from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.utils.monitor import monitor_cpu_usage
from task.passive_lv.fe_heart_sage_v1.data.datasets_preparation import (
FEHeartSageV1PreparationDataset, import_data_config
)

if __name__ == "__main__":
    monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(100,))
    monitor_thread.start()

    start_time = time.time()

    config = import_data_config("fe_heart_sage_v1")

    for data_type in [TRAIN_NAME, VALIDATION_NAME]:

        data_preprocess = FEHeartSageV1PreparationDataset(config, data_type)

    monitor_thread.join()

    print(f"{time.time() - start_time}s")
