import abc
import argparse
import os
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.train.callbacks.base_callback import CallbackList
from pkg.train.callbacks.model_checkpoint import ModelCheckpoint
from pkg.train.callbacks.tensorboard import TensorBoard
from pkg.train.config.base_config import BaseConfig
from pkg.train.datasets.base_datasets import BaseDataset
from pkg.train.model.base_model import BaseModule
from pkg.train.module.loss import EuclideanDistanceMSE
from pkg.utils.io import load_yaml
from pkg.utils.logging import init_logger
from pkg.utils.model_summary import summary

logger = init_logger("BASE_TRAINER")


class TrainerConfig(BaseConfig):
    """TrainerConfig class is inherent from BaseConfig class defining the structure for Trainer configuration."""

    def __init__(self) -> None:
        """Constructor to initialize a TrainerConfig object."""
        logger.info("=== Init Trainer Config ===")

        # parse args

        args = self.parse_args()
        repo_path: str = args.repo_path
        task_name: str = args.task_name

        # task base info
        task_path = f"{repo_path}/task/{task_name}"
        config_path = f"{task_path}/config/train_config.yaml"
        self.config: Dict = load_yaml(config_path)

        self.task_base = self.config["task_base"]
        self.task_name = self.task_base["task_name"]
        self.exp_name = self.task_base["exp_name"]

        self.task_base["repo_path"] = repo_path
        self.task_base["task_path"] = task_path
        self.task_base["config_path"] = config_path
        self.task_base["logs_base_path"] = f"{repo_path}/tmp/{task_name}/{self.exp_name}"

        self._create_logs_path()

        # task dataset info
        self.task_data = self.config.get("task_data", {})
        self.task_data["task_data_path"] = self.task_data.get(
            "task_data_path", f"{repo_path}/pkg/data/{self.task_name}"
        )

        # task trainer
        self.task_trainer = self.config["task_trainer"]

        # task train
        self.task_train = self.config["task_train"]

        logger.info(f"Data path: {self.task_data['task_data_path']}")

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train Model")

        parser.add_argument("--repo_path", type=str, help="current repo path")
        parser.add_argument("--task_name", type=str, help="task job name")

        args = parser.parse_args()

        return args

    def _create_logs_path(self) -> None:
        log_path = f"{self.task_base['repo_path']}/tmp"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        task_logs_path = f"{log_path}/{self.task_name}"
        if not os.path.exists(task_logs_path):
            os.makedirs(task_logs_path)

        exp_logs_path = f"{task_logs_path}/{self.exp_name}"
        if not os.path.exists(exp_logs_path):
            os.makedirs(exp_logs_path)
        elif not self.task_base.get("overwrite_exp_folder", True):
            raise ValueError(
                f"the current {exp_logs_path} directory can't be overwrite, please choice a new exp folder name"
            )

    def get_config(self) -> Dict:
        return {
            "task_base": self.task_base,
            "task_data": self.task_data,
            "task_trainer": self.task_trainer,
            "task_train": self.task_train,
        }


class BaseTrainer(abc.ABC):
    dataset_class: BaseDataset = BaseDataset

    def __init__(self, config: TrainerConfig):
        logger.info("====== Init Trainer ====== ")

        # === init config
        self.task_base: Dict = config.task_base
        self.task_data: Dict = config.task_data
        self.task_trainer: Dict = config.task_trainer
        self.task_train: Dict = config.task_train

        # === init tuning param
        self.loss: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # === init callback
        self.callback: Optional[CallbackList] = None

    # === dataset ===
    def create_dataset(self) -> (BaseDataset, BaseDataset):
        task_data = self.task_data

        train_dataset = self.dataset_class(task_data, TRAIN_NAME)
        logger.info(f"Number of train data points: {len(train_dataset)}")

        validation_dataset = self.dataset_class(task_data, VALIDATION_NAME)
        logger.info(f"Number of validation_data data points: {len(validation_dataset)}")

        return train_dataset, validation_dataset

    # === model ===
    def create_model(self) -> BaseModule:
        raise NotImplementedError("please implement create_model func")

    def print_model(self, model: nn.Module, inputs: List[List]):
        model_summary = self.task_trainer.get(
            "model_summary",
            {
                "show_input": False,
                "show_hierarchical": False,
                "print_summary": True,
                "max_depth": 999,
                "show_parent_layers": True,
            },
        )

        logger.info("=== Print Model Structure ===")
        logger.info(model)

        summary(
            model,
            inputs,
            show_input=model_summary["show_input"],
            show_hierarchical=model_summary["show_hierarchical"],
            print_summary=model_summary["print_summary"],
            max_depth=model_summary["max_depth"],
            show_parent_layers=model_summary["show_parent_layers"],
        )

    def create_optimize(self, model: nn.Module) -> None:
        optimizer_param = self.task_trainer["optimizer_param"]

        if optimizer_param["name"] == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_param["learning_rate"])
        else:
            raise ValueError(f"optimizer name do not set properly, please check: {optimizer_param['optimizer']}")

    def create_loss(self) -> None:
        loss_param = self.task_trainer["loss_param"]

        if loss_param["name"] == "mse":
            self.loss = nn.MSELoss()
        if loss_param["name"] == "euclidean_distance_mse":
            self.loss = EuclideanDistanceMSE()
        else:
            raise ValueError(f"loss name do not set properly, please check: {loss_param['name']}")

    def create_callback(self, model: nn.Module) -> None:
        callback_param = self.task_trainer["callback_param"]

        tensorboard_param = callback_param.get("tensorboard", {})
        tensorboard_param = self._check_callback_dir_exist(tensorboard_param)
        tensorboard = TensorBoard(tensorboard_param["log_dir"])

        checkpoint_param = callback_param.get("model_checkpoint", {})
        checkpoint_param = self._check_callback_dir_exist(checkpoint_param)
        model_checkpoint = ModelCheckpoint(
            checkpoint_param["log_dir"], save_freq=checkpoint_param.get("save_freq", "epoch")
        )

        self.callback = CallbackList(
            callbacks=[tensorboard, model_checkpoint],
            model=model,
        )

    def _check_callback_dir_exist(self, params: Dict) -> Dict:
        if "log_dir" not in params:
            params["log_dir"] = self.task_base["logs_base_path"]

        return params

    # === train ===
    def train(self):
        task_trainer = self.task_trainer

        # ====== Params ======
        epoch = task_trainer["epochs"]
        batch_size = task_trainer["batch_size"]
        shuffle = task_trainer["dataset_shuffle"]

        # ====== Generate dataset ======
        train_dataset, validation_dataset = self.create_dataset()

        # ====== Generate dataloder ======
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size)

        # ====== Create model ======
        model = self.create_model()

        self.print_model(model, train_dataset.get_head_inputs())

        # ====== Init optimizer ======
        self.create_optimize(model)

        # ====== Init loss ======
        self.create_loss()

        # ====== Init callback ======
        self.create_callback(model)

        self.callback.on_train_begin()

        for t in range(epoch):
            train_metrics = self.train_step(model, train_data_loader)

            val_metrics = self.validation_step(model, validation_data_loader)

            self.callback.on_epoch_end(t, train_metrics=train_metrics, val_metrics=val_metrics)

        self.callback.on_train_end(epoch=epoch)

    def compute_loss(self, predictions, labels):
        return self.loss(predictions, labels)

    def train_step(self, model: nn.Module, dataloder: DataLoader) -> Dict:
        task_trainer = self.task_trainer

        batch_size = task_trainer["batch_size"]
        train_loss = 0
        data_size = 0

        model.train()

        for batch, data in enumerate(dataloder):
            # Forward pass: compute predicted y by passing x to the model.
            # note: by default, we assume batch size = 1
            train_inputs, train_labels = data

            # Compute and print loss.
            outputs = model(train_inputs)
            loss = self.compute_loss(outputs, train_labels)

            train_loss += loss.item()
            data_size += batch_size

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            self.optimizer.step()

        metrics = {"loss": train_loss / data_size}

        return metrics

    def compute_validation_loss(self, predictions, labels):
        return self.compute_loss(predictions, labels)

    def validation_step(self, model: nn.Module, dataloder: DataLoader) -> Dict:
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        task_trainer = self.task_trainer

        batch_size = task_trainer["batch_size"]
        data_size = 0

        model.eval()

        val_loss = 0
        with torch.no_grad():
            for batch, val_data in enumerate(dataloder):
                val_inputs, val_labels = val_data

                outputs = model(val_inputs)

                val_loss += self.compute_validation_loss(outputs, val_labels).item()
                data_size += batch_size

        metrics = {"loss": val_loss / data_size}

        return metrics
