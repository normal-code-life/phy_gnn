from typing import Dict, Type
from pkg.train.config.base_config import BaseConfig
from pkg.utils.logging import init_logger
from torch import nn


logger = init_logger("BASE_MODEL")


class BaseModule(nn.Module, BaseConfig):
    def __init__(self, config: Dict, *args, **kwargs) -> None:

        self.prefix_name = "base_module"
        if "prefix_name" in config:
            self.prefix_name = config["prefix_name"]
        elif "prefix_name" in kwargs:
            self.prefix_name = kwargs.pop("prefix_name")

        super(BaseModule, self).__init__(*args, **kwargs)

    def _init_graph(self) -> None:
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required '_init_graph' function")

    def get_config(self) -> Dict:
        return {
            "prefix_name": self.prefix_name,
        }

