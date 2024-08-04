from typing import Dict, Optional

import torch
from tensorboardX import SummaryWriter

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logging import init_logger


class TensorBoardCallback(CallBack):
    def __init__(self, task_base_param: Dict, param: Dict) -> None:
        super(TensorBoardCallback, self).__init__(task_base_param, param)

        use_profiler: bool = param["profiler"]

        self.writer = SummaryWriter(self.log_dir)

        self.profiler: Optional[torch.profiler.profile] = None
        if use_profiler:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{self.log_dir}/profiler"),
                record_shapes=True,
                with_stack=True,
            )

        self.logger = init_logger("TENSORBOARD_CALLBACK")

    def on_train_begin(self):
        self.logger.info("====== model training start ======")

        if self.profiler:
            self.profiler.start()

    def on_train_end(self, **kwargs):
        self.logger.info("====== model training end ======")

        if self.profiler:
            self.profiler.stop()

        self.writer.close()

    def on_epoch_begin(self, epoch, **kwargs):
        if self.profiler:
            self.profiler.step()

    def on_epoch_end(self, epoch, **kwargs):
        if "train_metrics" in kwargs:
            train_metrics = kwargs["train_metrics"]

            for key, value in train_metrics.items():
                self.writer.add_scalar(f"{TRAIN_NAME}/{key}", value, epoch)


        if "val_metrics" in kwargs:
            val_metrics = kwargs["val_metrics"]

            for key, value in val_metrics.items():
                self.writer.add_scalar(f"{VALIDATION_NAME}/{key}", value, epoch)

