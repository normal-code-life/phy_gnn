from typing import Dict

import torch

from pkg.train.model.base_model import BaseModule


class PoolingLayer(BaseModule):
    """Base class for pooling layers.

    Implements common functionality for aggregating features along a dimension.

    Args:
        config (Dict): Configuration dictionary containing:
            - agg_dim (int): Dimension to aggregate along (default: -1)
            - keep_dim (bool): Whether to keep the aggregated dimension (default: True)
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments

    Attributes:
        agg_dim (int): Dimension to aggregate along
        keep_dim (bool): Whether to keep the aggregated dimension
    """

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super(PoolingLayer, self).__init__(config, *args, **kwargs)

        self.agg_dim = config.get("agg_dim", -1)
        self.keep_dim = config.get("keep_dim", True)


class MeanAggregator(PoolingLayer):
    """Mean pooling aggregator.

    Aggregates features by taking the mean along a specified dimension.

    Args:
        config (Dict): Configuration dictionary containing:
            - agg_dim (int): Dimension to aggregate along (default: -1)
            - keep_dim (bool): Whether to keep the aggregated dimension (default: True)
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
    """

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
    """Sum pooling aggregator.

    Aggregates features by taking the sum along a specified dimension.

    Args:
        config (Dict): Configuration dictionary containing:
            - agg_dim (int): Dimension to aggregate along (default: -1)
            - keep_dim (bool): Whether to keep the aggregated dimension (default: True)
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
    """

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
