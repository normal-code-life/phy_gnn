import abc
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from pkg.train.model.base_model import BaseModule
from pkg.utils.logs import init_logger

logger = init_logger("mlp_layer_ln")


class MLPLayerBase(BaseModule):
    def __init__(self, config: Dict, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.layer_name = f"{self.prefix_name}_mlp"

        self.unit_sizes: List[int] = []
        if isinstance(config["unit_sizes"], list):
            # WARN: layer_sizes should contain the first input layer and final layer's output dim
            self.unit_sizes = config["unit_sizes"]
        else:
            raise ValueError("the 'unit_sizes' should be a list, and should contain the final layer's output size")

        self.batch_norm = config.get("batch_norm", False)
        self.layer_norm = config.get("layer_norm", False)
        self.activation = config.get("activation", None)  # by default, the last layer will not have the activation func
        self.init_func = config.get("init_func", "xavier_uniform")
        self.init_weight_file_path = config.get("init_weight_file_path", None)  # if not None, weight will be assigned

        self.mlp_layers: nn.Sequential = nn.Sequential()
        self._init_graph()

    def get_config(self) -> Dict:
        base_config = super().get_config()

        mlp_config = {
            "unit_sizes": self.unit_sizes,
            "layer_name": self.layer_name,
            "batch_norm": self.batch_norm,
            "layer_norm": self.layer_norm,
            "activation": self.activation,
            "init_weight_file_path": self.init_weight_file_path,
        }

        return {**base_config, **mlp_config}

    @property
    def get_prefix_name(self) -> str:
        return self.prefix_name

    @abc.abstractmethod
    def _init_graph(self) -> None:
        raise NotImplementedError("please implement this method")

    def _init_fc(
        self,
        cur_layer_name: str,
        input_unit_size: int,
        output_unit_size: int,
    ) -> None:
        fc = nn.Linear(input_unit_size, output_unit_size)

        if self.init_weight_file_path:
            self.weight_init_dict: Dict = dict()
            with open(self.init_weight_file_path, "rb") as file:
                self.weight_init_dict = np.load(file, allow_pickle=True).item()
            if cur_layer_name in self.weight_init_dict:
                fc.weight = nn.Parameter(torch.tensor(self.weight_init_dict[cur_layer_name]).t())
                logger.info(f"init {cur_layer_name} model layer from {self.init_weight_file_path}")
            else:
                raise ValueError(f"error, we don't have this layer {cur_layer_name}")
        else:
            if self.init_func == "xavier_uniform":
                nn.init.xavier_uniform_(fc.weight)
            elif self.init_func == "xavier_normal":
                nn.init.xavier_normal_(fc.weight)
            else:
                raise Exception(f"please define the init_func correctly, currently init_func={self.init_func}")

        nn.init.zeros_(fc.bias)
        self.mlp_layers.add_module(cur_layer_name, fc)

    def forward(self, x):
        return self.mlp_layers(x)
