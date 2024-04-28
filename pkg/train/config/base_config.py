import abc
from typing import Dict


class BaseConfig(abc.ABC):
    """BaseConfig class is an abstract base class defining the basic structure for configuration classes."""

    def get_config(self) -> Dict:
        raise NotImplementedError("please implement 'get_config' method")
