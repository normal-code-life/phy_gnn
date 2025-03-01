from typing import Dict, Optional

import torch
from tensorboardX import SummaryWriter

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logs import init_logger


class TensorBoardCallback(CallBack):
    """A callback for logging training metrics and profiling information to TensorBoard.

    This callback handles:
    - Writing training and validation metrics to TensorBoard
    - Optional profiling of model training using PyTorch profiler
    - Proper cleanup of TensorBoard writer and profiler resources

    Parameters:
    -----------
    task_base_param : Dict
        Dictionary containing base parameters like log directory path.

    param : Dict
        Dictionary containing profiler configuration parameters:
        - profiler (bool): Whether to enable profiling
        - wait (int): Number of steps to wait before profiling starts
        - warmup (int): Number of warmup steps for profiler
        - active (int): Number of active profiling steps
        - repeat (int): Number of times to repeat profiling cycle

    Attributes:
    -----------
    writer : SummaryWriter
        TensorBoard writer instance for logging metrics.

    profiler : Optional[torch.profiler.profile]
        PyTorch profiler instance if profiling is enabled.

    logger : Logger
        Logger instance for callback-specific logging.
    """

    def __init__(self, task_base_param: Dict, param: Dict) -> None:
        super(TensorBoardCallback, self).__init__(task_base_param, param)

        use_profiler: bool = param["profiler"]
        profiler_wait: int = param.get("wait", 1)
        profiler_warmup: int = param.get("warmup", 1)
        profiler_active: int = param.get("active", 1)
        profiler_repeat: int = param.get("repeat", 1)

        self.writer = SummaryWriter(self.log_dir)

        self.profiler: Optional[torch.profiler.profile] = None
        if use_profiler:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=profiler_wait, warmup=profiler_warmup, active=profiler_active, repeat=profiler_repeat
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{self.log_dir}/profiler"),
                record_shapes=True,
                with_stack=True,
            )

        self.logger = init_logger("TENSORBOARD_CALLBACK")

    def on_train_begin(self):
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
