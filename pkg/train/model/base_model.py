from typing import Dict
from pkg.train.config.base_config import BaseConfig
from pkg.utils.io import load_yaml
from pkg.utils.logging import init_logger
from pkg.utils import io
from torch import nn


logger = init_logger("BASE_MODEL")


class ModelConfig(BaseConfig):
    def __init__(self, config: Dict):
        self.config = config

    def get_config(self):
        return self.config


class BaseModel(nn.Module):
    def __init__(self, config: ModelConfig, *args, **kwargs) -> None:
        super(BaseModel, self).__init__(*args, **kwargs)
        self.config = config

    def _init_graph(self):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required '_init_graph' function")

