import os
from typing import Union

import torch

from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logging import init_logger


class ModelCheckpoint(CallBack):
    def __init__(self, log_dir: str, save_freq: Union[int, str]) -> None:
        self.checkpoint_dir = self._check_and_create_folder(log_dir, "checkpoint")
        self.model_dir = self._check_and_create_folder(log_dir, "model")

        self.save_freq = save_freq

        self.logger = init_logger("MODEL_CHECKPOINT")

    @staticmethod
    def _check_and_create_folder(log_dir: str, folder_name: str) -> str:
        path = f"{log_dir}/{folder_name}"

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def on_train_end(self, epoch, **kwargs):
        self.logger.info("========= model saving =========")

        self.save_model()
        self.save_checkpoint(epoch)

    def save_model(self):
        torch.save(self.model, f"{self.model_dir}/model.pth")

    def save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/ckpt_{epoch}.pth")

    def on_epoch_end(self, epoch, **kwargs):
        self.logger.info("========= checkpoint saving =========")

        if isinstance(self.save_freq, str) and self.save_freq == "epoch":
            self.save_checkpoint(epoch)
        elif isinstance(self.save_freq, int) and epoch % self.save_freq == 0:
            self.save_checkpoint(epoch)
