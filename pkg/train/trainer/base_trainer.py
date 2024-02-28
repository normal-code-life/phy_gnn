import abc
from typing import Dict
from pkg.train.config.base_config import BaseConfig
from pkg.utils.io import load_yaml
from pkg.utils.logging import init_logger
from pkg.utils import io


logger = init_logger("BASE_TRAINER")


class TrainerConfig(BaseConfig):
    """
    TrainerConfig class is inherent from BaseConfig class defining the structure for Trainer configuration classes.

    Attributes:
        repo_root_path: Attribute to store the repo default root path.
        config_path: Attribute to store configuration information, loaded using the yaml.unsafe_load method.
    """

    def __init__(self, task_path: str, config_path: str):
        """
        Constructor to initialize a TrainerConfig object.

        Args:
            task_path (str): String containing default repo root path.
            config_path (str): String containing configuration information.

        Notes:
            1. Use the yaml.unsafe_load method to load configuration information.
        """
        self.config: Dict = load_yaml(config_path)

        # task base info
        self.task_base = self.config["task_base"]
        self.task_name = self.task_base["task_name"]

        repo_root_path = io.get_repo_path(task_path)
        self.task_base["repo_root_path"] = repo_root_path
        self.task_base["task_path"] = task_path

        # task dataset info
        self.task_data = self.config.get("task_data", {})
        self.task_data["task_data_path"] = self.task_data.get(
            "task_data_path", f"{repo_root_path}/pkg/data/{self.task_name}"
        )

        # task trainer
        self.task_trainer = self.config["task_trainer"]

        # task train
        self.task_train = self.config["task_train"]

    def get_config(self):
        return {
            "task_base": self.task_base,
            "task_data": self.task_data,
            "task_trainer": self.task_trainer,
            "task_train": self.task_train,
        }


class BaseTrainer(abc.ABC):
    def __init__(self, config: TrainerConfig):
        logger.info("====== Trainer Init ====== ")

        self.task_base = config.task_base
        self.task_data = config.task_data
        self.task_trainer = config.task_trainer
        self.task_train = config.task_train
