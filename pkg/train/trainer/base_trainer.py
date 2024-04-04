import abc
import time
from typing import Dict, List

import torch
from pytorch_model_summary import summary
from torch import nn
from torch.utils.data import DataLoader

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.train.config.base_config import BaseConfig
from pkg.train.datasets.base_datasets import BaseDataset
from pkg.train.model.base_model import BaseModule
from pkg.train.module.loss import EuclideanDistanceMSE
from pkg.utils import io
from pkg.utils.io import load_yaml
from pkg.utils.logging import init_logger

logger = init_logger("BASE_TRAINER")


class TrainerConfig(BaseConfig):
    """
    TrainerConfig class is inherent from BaseConfig class defining the structure for Trainer configuration classes.

    Attributes:
        repo_root_path: Attribute to store the repo default root path.
        config_path: Attribute to store configuration information, loaded using the yaml.unsafe_load method.
    """

    def __init__(self, config_path: str):
        """
        Constructor to initialize a TrainerConfig object.

        Args:
            task_path (str): String containing default repo root path.
            config_path (str): String containing configuration information.

        Notes:
            1. Use the yaml.unsafe_load method to load configuration information.
        """
        logger.info("=== Init Trainer Config ===")
        self.config: Dict = load_yaml(config_path)

        # task base info
        self.task_base = self.config["task_base"]
        self.task_name = self.task_base["task_name"]

        repo_root_path = io.get_repo_path(config_path)
        self.task_base["repo_root_path"] = repo_root_path
        self.task_base["config_path"] = config_path

        # task dataset info
        self.task_data = self.config.get("task_data", {})
        self.task_data["task_data_path"] = self.task_data.get(
            "task_data_path", f"{repo_root_path}/pkg/data/{self.task_name}"
        )

        # task trainer
        self.task_trainer = self.config["task_trainer"]

        # task train
        self.task_train = self.config["task_train"]

        logger.info(f"Data path: {self.task_data['task_data_path']}")

    def get_config(self):
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

        self.task_base = config.task_base
        self.task_data = config.task_data
        self.task_trainer = config.task_trainer
        self.task_train = config.task_train

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
        logger.info("=== Print Model Structure ===")
        logger.info(model)

        summary(
            model,
            inputs,
            show_input=True,
            # show_hierarchical=True,
            print_summary=True,
            max_depth=999,
            show_parent_layers=True,
        )

    def create_optimize(self, model: nn.Module) -> None:
        optimizer_param = self.task_trainer["optimizer_param"]

        self.optimizer: torch.optim.Optimizer
        if optimizer_param["name"] == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_param["learning_rate"])
        else:
            raise ValueError(f"optimizer name do not set properly, please check: {optimizer_param['optimizer']}")

    def create_loss(self) -> None:
        loss_param = self.task_trainer["loss_param"]

        self.loss: nn.Module
        if loss_param["name"] == "mse":
            self.loss = nn.MSELoss()
        if loss_param["name"] == "euclidean_distance_mse":
            self.loss = EuclideanDistanceMSE()
        else:
            raise ValueError(f"loss name do not set properly, please check: {loss_param['name']}")

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

        logger.info("====== model training start! ======")

        start_time = time.time()

        for t in range(epoch):
            train_metrics = self.train_step(model, train_data_loader)
            val_metrics = self.validation_step(model, validation_data_loader)

            logger.info(
                "epoch: %d, training time: %ds, train_loss: %f, val_loss: %f",
                t,
                time.time() - start_time,
                train_metrics["loss"],
                val_metrics["loss"],
            )

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
