import os
from typing import Union, Dict

import torch

from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logging import init_logger


class ModelCheckpointCallback(CallBack):
    def __init__(self, task_base_param: Dict, param: Dict) -> None:
        super(ModelCheckpointCallback, self).__init__(task_base_param, param)
        self.checkpoint_dir = self._check_and_create_folder(self.log_dir, "checkpoint")
        self.model_dir = self._check_and_create_folder(self.log_dir, "model")

        self.save_freq = param.get("save_freq", "epoch")

        self.logger = init_logger("MODEL_CHECKPOINT")

    @staticmethod
    def _check_and_create_folder(log_dir: str, folder_name: str) -> str:
        path = f"{log_dir}/{folder_name}"

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def on_train_end(self, epoch, **kwargs):
        self.logger.info("========= model saving =========")

        # self.save_model()
        self.save_checkpoint(epoch)

    def save_model(self):
        torch.save(self.model, f"{self.model_dir}/model.pth")

    def save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/ckpt_{epoch}.pth")

    def on_epoch_end(self, epoch, **kwargs):
        save_checkpoint = False

        if (
            isinstance(self.save_freq, str)
            and self.save_freq == "epoch"
            or isinstance(self.save_freq, int)
            and epoch % self.save_freq == 0
        ):
            save_checkpoint = True

        if save_checkpoint:
            # self.logger.info(f"========= checkpoint saving epoch={epoch} =========")
            self.save_checkpoint(epoch)
