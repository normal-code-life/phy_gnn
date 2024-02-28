import torch.nn as nn
from typing import Dict, List
from pkg.utils.logging import init_logger
from pkg.train.model.base_model import BaseModuleConfig, BaseModule
from pkg.train.module.activation import get_activation

logger = init_logger("mlp_layer")


class MLPConfig(BaseModuleConfig):
    def __init__(self, config: Dict, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.layer_name = "mlp"

        self.unit_sizes: List[int] = []
        if isinstance(config["unit_sizes"], list):
            logger.info("WARN: layer_sizes should contain the final layer's output dim, or it will panic")
            self.unit_sizes = config["unit_sizes"]
        else:
            raise ValueError("the 'unit_sizes' should be a list, and should contain the final layer's output size")

        self.batch_norm = config.get("batch_norm", False)
        self.layer_norm = config.get("layer_norm", False)
        self.activation = config.get("activation", None)  # by default, the last layer will not have the activation func

    def get_config(self):
        base_config = super().get_config()

        mlp_config = {
            "unit_sizes": self.unit_sizes,
            "layer_name": self.layer_name,
            "batch_norm": self.batch_norm,
            "layer_norm": self.layer_norm,
            "activation": self.activation,
        }

        return {**base_config, **mlp_config}


class MLPModule(BaseModule):
    def __init__(self, config: MLPConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.mlp_layers = self._init_graph(config)

    def _init_graph(self, config: MLPConfig):
        sequential = nn.Sequential()

        for i in range(len(config.unit_sizes) - 1):
            cur_layer_name = f"{config.prefix_name}_{config.layer_name}_l{i + 1}"

            # add fc layer
            sequential.add_module(cur_layer_name, nn.Linear(config.unit_sizes[i], config.unit_sizes[i + 1]))

            # add batch/layer norm
            if config.batch_norm:
                sequential.add_module(f"{cur_layer_name}_bn", nn.BatchNorm1d(config.unit_sizes[i + 1]))
            elif config.layer_norm:
                sequential.add_module(f"{cur_layer_name}_ln", nn.LayerNorm(config.unit_sizes[i + 1]))

            # add activation
            if config.activation and i != len(config.unit_sizes) - 2:
                sequential.add_module(f"{cur_layer_name}_ac", get_activation(config.activation))

        return sequential

    def forward(self, x):
        return self.mlp_layers(x)
