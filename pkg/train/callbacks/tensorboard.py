import time

from tensorboardX import SummaryWriter

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logging import init_logger


class TensorBoard(CallBack):
    def __init__(self, log_dir: str) -> None:
        log_dir = f"{log_dir}/logs"

        self.writer = SummaryWriter(log_dir)

        self.start_time = time.time()

        self.logger = init_logger("TENSORBOARD")

    def on_train_begin(self):
        self.logger.info("====== model training start ======")

    def on_train_end(self, **kwargs):
        self.logger.info("====== model training end ======")
        self.writer.close()

    def on_epoch_end(self, epoch, **kwargs):
        if "train_metrics" in kwargs:
            train_metrics = kwargs["train_metrics"]
        else:
            raise ValueError("please feed train_metrics")

        if "val_metrics" in kwargs:
            val_metrics = kwargs["val_metrics"]
        else:
            raise ValueError("please feed val_metrics")

        for key, value in train_metrics.items():
            self.writer.add_scalar(f"{TRAIN_NAME}/{key}", value, epoch)

        for key, value in val_metrics.items():
            self.writer.add_scalar(f"{VALIDATION_NAME}/{key}", value, epoch)

        self.logger.info(
            "epoch: %d, training time: %ds, train_loss: %f, val_loss: %f",
            epoch,
            time.time() - self.start_time,
            train_metrics["loss"],
            val_metrics["loss"],
        )
