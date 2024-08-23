import os
from typing import Dict, Optional

import torch
import shutil
from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logs import init_logger

logger = init_logger("MODEL_CHECKPOINT")


class ModelCheckpointCallback(CallBack):
    def __init__(self, task_base_param: Dict, param: Dict) -> None:
        super(ModelCheckpointCallback, self).__init__(task_base_param, param)
        self.checkpoint_dir = self._check_and_create_folder(self.log_dir, "checkpoint")
        self.model_dir = self._check_and_create_folder(self.log_dir, "model")

        self.save_freq = param.get("save_freq", "epoch")

        self.save_model_freq = param.get("save_model_freq", None)

    @staticmethod
    def _check_and_create_folder(log_dir: str, folder_name: str) -> str:
        path = f"{log_dir}/{folder_name}"

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def on_train_end(self, epoch: int, **kwargs):
        logger.info("========= final model saving =========")

        self.save_checkpoint(epoch, **kwargs)
        self.save_model()

    def on_epoch_end(self, epoch: int, **kwargs):
        if epoch == 0:
            return

        save_checkpoint = False
        save_model = False

        if isinstance(self.save_freq, str) and self.save_freq == "epoch":
            save_checkpoint = True
        elif isinstance(self.save_freq, int) and epoch % self.save_freq == 0:
            save_checkpoint = True

        if self.save_model_freq and epoch % self.save_model_freq == 0:
            save_model = True

        if save_checkpoint:
            self.save_checkpoint(epoch, **kwargs)

        if save_model:
            self.save_model(epoch)

    def save_checkpoint(self, epoch: int, **kwargs) -> None:
        ckpt_dir = f"{self.checkpoint_dir}/ckpt_{epoch}.pth"

        save_ckpt_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }

        if "train_metrics" in kwargs:
            save_ckpt_dict.update(kwargs["train_metrics"])

        if "val_metrics" in kwargs:
            save_ckpt_dict.update(kwargs["val_metrics"])

        torch.save(save_ckpt_dict, ckpt_dir)
        shutil.copy(ckpt_dir, f"{self.checkpoint_dir}/ckpt.pth")

        logger.info(f"saving ckpt epoch={epoch} to {ckpt_dir} success")

    def save_model(self, epoch: Optional[int] = None) -> None:
        model_dir = f"{self.model_dir}/model.pth" if epoch is None else f"{self.model_dir}/model_{epoch}.pth"

        torch.save(self.model, f"{model_dir}")

        logger.info(f"saving model epoch={epoch} to {model_dir} success")

