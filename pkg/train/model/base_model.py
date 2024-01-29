from typing import Dict
from pkg.train.config.base_config import BaseConfig
from pkg.utils.io import load_yaml
from pkg.utils.logging import init_logger
from pkg.utils import io
from torch import nn


logger = init_logger("BASE_MODEL")


class BaseModuleConfig(BaseConfig):
    def __init__(self, config: Dict, **kwargs) -> None:
        self.prefix_name = "base_module"
        if "prefix_name" in config:
            self.prefix_name = config["prefix_name"]
        elif "prefix_name" in kwargs:
            self.prefix_name = kwargs["prefix_name"]

        self.config = config

    def get_config(self):
        return {
            "prefix_name": self.prefix_name,
        }


class BaseModule(nn.Module):
    def __init__(self, config: BaseModuleConfig, *args, **kwargs) -> None:
        super(BaseModule, self).__init__(*args, **kwargs)
        self.config = config

    def _init_graph(self):
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required '_init_graph' function")

