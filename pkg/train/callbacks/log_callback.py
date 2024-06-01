import os
import time
from typing import Dict, Optional, Union
from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logging import init_logger


class LogCallback(CallBack):
    def __init__(self, task_base_param: Dict, param: Dict) -> None:
        super(LogCallback, self).__init__(task_base_param, param)

        self.config_path = task_base_param["config_path"]

        self.start_time = time.time()

        self.update_freq = param.get("update_freq", "epoch")

        self.save_config = param.get("save_config", False)

        self.save_task_code = param.get("save_task_code", False)

        self.logger = init_logger("LOGS_CALLBACK")

    def on_train_begin(self, **kwargs):
        if self.save_config:
            cmd = f"cp {self.config_path} {self.log_dir}/"
            self.logger.info(f"execute {cmd}")
            os.system(cmd)

        if self.save_task_code:
            cmd = f"cp -r {self.task_dir} {self.log_dir}/code/"
            self.logger.info(f"execute {cmd}")
            os.system(cmd)

    def on_epoch_end(self, epoch, **kwargs):
        if "train_metrics" in kwargs:
            train_metrics = kwargs["train_metrics"]
        else:
            raise ValueError("please feed train_metrics")

        if "val_metrics" in kwargs:
            val_metrics = kwargs["val_metrics"]
        else:
            raise ValueError("please feed val_metrics")

        if self.update_freq == "epoch" or epoch % self.update_freq == 0:
            train_logs = {
                **{name: val for name, val in train_metrics.items()},
                **{name: val for name, val in val_metrics.items()}
            }
            msg = " - ".join(f"{name}: {round(val, 5)}" for name, val in train_logs.items())
            self.logger.info(f"metrics: {epoch} - {round(time.time() - self.start_time, 2)}s - {msg}")
