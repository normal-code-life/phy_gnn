import abc
import argparse
import os
from typing import Dict, List, Optional, Union

import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.train.callbacks.base_callback import CallbackList
from pkg.train.callbacks.log_callback import LogCallback
from pkg.train.callbacks.model_checkpoint_callback import ModelCheckpointCallback
from pkg.train.callbacks.tensorboard_callback import TensorBoardCallback
from pkg.train.config.base_config import BaseConfig
from pkg.train.datasets.base_datasets import BaseAbstractDataset, BaseDataset
from pkg.train.model.base_model import BaseModule
from pkg.train.module.loss import EuclideanDistanceMSE
from pkg.utils.io import load_yaml
from pkg.utils.logging import init_logger
from pkg.utils.model_summary import summary
from pkg.train.datasets.shuffle_iterable_datasets import ShuffledIterableDataset

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
        config_name: str = args.config_name

        # task base info
        task_path = f"{repo_path}/task/{task_name}"
        config_path = f"{task_path}/config/{config_name}.yaml"
        self.config: Dict = load_yaml(config_path)

        self.task_base = self.config["task_base"]
        self.task_name = self.task_base["task_name"]
        self.exp_name = self.task_base["exp_name"]

        self.task_base["repo_path"] = repo_path
        self.task_base["task_path"] = task_path
        self.task_base["config_path"] = config_path
        self.task_base["logs_base_path"] = f"{repo_path}/tmp/{task_name}/{self.exp_name}"

        self.task_base["gpu"] = self.task_base["gpu"] and torch.cuda.is_available()
        self.task_base["cuda_core"] = self.task_base.get("cuda_core", "cuda:0")

        if self.task_base["gpu"]:
            torch.cuda.set_device(self.task_base["cuda_core"])

        self._create_logs_path()

        # task dataset info
        self.task_data = self.config.get("task_data", {})
        self.task_data["task_path"] = task_path
        task_data_name = self.task_data.get("task_data_name", self.task_name)
        self.task_data["task_data_path"] = self.task_data.get(
            "task_data_path", f"{repo_path}/pkg/data/{task_data_name}"
        )
        self.task_data["gpu"] = self.task_base["gpu"]

        # task trainer
        self.task_trainer = self.config["task_trainer"]
        self.task_trainer["gpu"] = self.task_base["gpu"]

        # task train
        self.task_train = self.config["task_train"]

        logger.info(f"Data path: {self.task_data['task_data_path']}")

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train Model")

        parser.add_argument("--repo_path", type=str, help="current repo path")
        parser.add_argument("--task_name", type=str, help="task job name")
        parser.add_argument("--config_name", default="train_config", type=str, help="config file name")

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

        logger.info(f"{exp_logs_path} setup done")

    def get_config(self) -> Dict:
        return {
            "task_base": self.task_base,
            "task_data": self.task_data,
            "task_trainer": self.task_trainer,
            "task_train": self.task_train,
        }


class BaseTrainer(abc.ABC):
    dataset_class: BaseAbstractDataset = BaseAbstractDataset

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
        self.metrics: Dict[str, callable] = {}

        # === init callback
        self.callback: Optional[CallbackList] = None

        # === init others
        self.gpu: Union[bool, int] = self.task_trainer.get("gpu", False)
        self.static_graph: bool = self.task_trainer.get("static_graph", False)

    # === dataset ===
    def create_dataset(self) -> (BaseAbstractDataset, BaseAbstractDataset):
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

    def create_metrics(self):
        metrics_param = self.task_trainer["metrics_param"]
        for p in metrics_param:
            if p == "mean_absolute_error":
                self.metrics["mean_absolute_error"] = torchmetrics.functional.mean_absolute_error
            elif p == "explained_variance":
                self.metrics["explained_variance"] = torchmetrics.functional.explained_variance
            else:
                raise ValueError(f"metrics name do not set properly, please check: {p}")

    def create_callback(self, model: nn.Module) -> None:
        callback_param = self.task_trainer["callback_param"]

        tensorboard_param = callback_param.get("tensorboard", {})
        tensorboard = TensorBoardCallback(self.task_base, tensorboard_param)

        checkpoint_param = callback_param.get("model_checkpoint", {})
        model_checkpoint = ModelCheckpointCallback(self.task_base, checkpoint_param)

        logs_param = callback_param.get("logs", {})
        log_checkpoint = LogCallback(self.task_base, logs_param)

        self.callback = CallbackList(
            callbacks=[tensorboard, model_checkpoint, log_checkpoint],
            model=model,
        )

    # === train ===
    def train(self):
        task_trainer = self.task_trainer

        # ====== Params ======
        epoch = task_trainer["epochs"]
        dataset_param = task_trainer["dataset_param"]

        # ====== Generate dataset ======
        train_dataset, validation_dataset = self.create_dataset()

        # ====== Generate dataloder ======
        if isinstance(train_dataset, BaseDataset):
            train_data_loader = DataLoader(
                dataset=train_dataset,
                batch_size=dataset_param.get("batch_size", 1),
                shuffle=dataset_param.get("train_shuffle", True),
                # num_workers=dataset_param.get("num_workers", 0),
                # prefetch_factor=dataset_param.get("prefetch_factor", None)
            )
            validation_data_loader = DataLoader(
                dataset=validation_dataset,
                batch_size=dataset_param.get("val_batch_size", len(validation_dataset)),
                shuffle=dataset_param.get("test_shuffle", False),
                # num_workers=dataset_param.get("num_workers", 0),
                # prefetch_factor=dataset_param.get("prefetch_factor", None)
            )
        else:
            train_dataset = ShuffledIterableDataset(train_dataset, dataset_param.get("shuffle_size", 1))

            train_data_loader = DataLoader(
                dataset=train_dataset,
                batch_size=dataset_param.get("batch_size", 1),
                # num_workers=dataset_param.get("num_workers", 0),
                # prefetch_factor=dataset_param.get("prefetch_factor", None)
            )

            validation_dataset = ShuffledIterableDataset(validation_dataset, dataset_param.get("shuffle_size", 1))

            validation_data_loader = DataLoader(
                dataset=validation_dataset,
                batch_size=dataset_param.get("val_batch_size", 1),
                # num_workers=dataset_param.get("num_workers", 0),
                # prefetch_factor=dataset_param.get("prefetch_factor", None)
            )

        # ====== Create model ======
        model = self.create_model()

        if self.gpu:
            model = model.cuda()
            logger.info(f"cuda version: {torch.version.cuda}")
            logger.info(f"model device check: {next(model.parameters()).device}")

        self.print_model(model, train_dataset.get_head_inputs(dataset_param.get("batch_size", 1)))

        if self.static_graph:
            model = torch.jit.trace(model, train_dataset.get_head_inputs(1))

        # ====== Init optimizer ======
        self.create_optimize(model)

        # ====== Init loss & metrics ======
        self.create_loss()
        self.create_metrics()

        # ====== Init callback ======
        self.create_callback(model)

        self.callback.on_train_begin()

        for t in range(epoch):
            self.callback.on_epoch_begin(t)

            train_metrics = self.train_step(model, train_data_loader)

            val_metrics = self.validation_step(model, validation_data_loader)

            self.callback.on_epoch_end(t, train_metrics=train_metrics, val_metrics=val_metrics)

        self.callback.on_train_end(epoch=epoch)

    def compute_loss(self, predictions: torch.Tensor, labels: torch.Tensor):
        return self.loss(predictions, labels)

    def compute_metrics(self, metrics_func: callable, predictions: torch.Tensor, labels: torch.Tensor):
        return metrics_func(predictions, labels)

    def train_step(self, model: nn.Module, dataloder: DataLoader) -> Dict:
        batch_cnt = 0
        metrics = {}

        model.train()

        for batch, data in enumerate(dataloder):
            # Forward pass: compute predicted y by passing x to the model.
            # note: by default, we assume batch size = 1
            train_inputs, train_labels = data

            # Compute and print loss.
            outputs = model(train_inputs)
            loss = self.compute_loss(outputs, train_labels)

            batch_cnt += 1
            metrics["train_loss"] = metrics["train_loss"] + loss.item() if "train_loss" in metrics else loss.item()

            # print(
            # f"===> {batch},
            # {batch_size},
            # {loss},
            # {metrics['train_loss']},
            # {batch_cnt},
            # {metrics['train_loss'] / batch_cnt}"
            # )

            # Before the backward pass, use the optimizer object to zero all the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            self.optimizer.step()

        metrics = {"train_loss": metrics["train_loss"] / batch_cnt}

        return metrics

    def compute_validation_loss(self, predictions, labels):
        return self.compute_loss(predictions, labels)

    def validation_step(self, model: nn.Module, dataloder: DataLoader) -> Dict:
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        batch_cnt = 0
        metrics = {}

        model.eval()

        with torch.no_grad():
            for batch, val_data in enumerate(dataloder):
                batch_cnt += 1

                val_inputs, val_labels = val_data

                outputs = model(val_inputs)

                loss = self.compute_validation_loss(outputs, val_labels)

                metrics["val_loss"] = metrics["val_loss"] + loss.item() if "val_loss" in metrics else loss.item()

                # Compute metrics
                for p in self.metrics:
                    results = self.compute_metrics(self.metrics[p], outputs, val_labels)
                    metrics[f"val_{p}"] = metrics[f"val_{p}"] + results.item() if p in metrics else results.item()

        for p in metrics:
            metrics[p] = metrics[p] / batch_cnt

        return metrics
