from typing import Dict

import torch

from pkg.train.model.base_model import BaseModule


class PoolingLayer(BaseModule):
    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super(PoolingLayer, self).__init__(config, *args, **kwargs)

        self.agg_dim = config.get("agg_dim", -1)
        self.keep_dim = config.get("keep_dim", True)


class MeanAggregator(PoolingLayer):
    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super(MeanAggregator, self).__init__(config, *args, **kwargs)
        config["prefix_name"] = "mean_pooling_agg"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean aggregator.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        return torch.mean(x, dim=self.agg_dim, keepdim=self.keep_dim)


class SUMAggregator(PoolingLayer):
    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super(SUMAggregator, self).__init__(config, *args, **kwargs)
        self.prefix_name = "sum_pooling_agg"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum aggregator.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        return torch.sum(x, dim=self.agg_dim, keepdim=self.keep_dim)
