import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from pkg.train.model.base_model import ModelConfig, BaseModel


class MLPConfig(ModelConfig):
    def __init__(self, config: Dict) -> None:



    def get_config(self):
        pass


class MLP(BaseModel):
    def __init__(self, config: MLPConfig, *args, **kwargs) -> None:
        super().__init__(config, args, kwargs)


    def _init_graph(self):
        pass

    def forward(self, x):
        pass





