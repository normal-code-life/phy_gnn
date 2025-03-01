from typing import Dict

from torch import nn

from pkg.train.config.base_config import BaseConfig
from pkg.utils.logs import init_logger

logger = init_logger("BASE_MODEL")


class BaseModule(nn.Module, BaseConfig):
    """Base module class for neural network models.

    Provides core functionality for model configuration and initialization.
    Subclasses implement specific model architectures.

    Args:
        config (Dict): Configuration dictionary containing model parameters
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments

    Attributes:
        prefix_name (str): Prefix for model component names
    """

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        self.prefix_name = "base_module"
        if "prefix_name" in config:
            self.prefix_name = config["prefix_name"]
        elif "prefix_name" in kwargs:
            self.prefix_name = kwargs.pop("prefix_name")

        super(BaseModule, self).__init__(*args, **kwargs)

    def _init_graph(self) -> None:
        """Initialize the model computation graph.

        Must be implemented by subclasses to define model architecture.

        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required '_init_graph' function")

    def get_config(self) -> Dict:
        """Get model configuration.

        Returns:
            Dict: Configuration dictionary containing model parameters
        """
        return {
            "prefix_name": self.prefix_name,
        }
