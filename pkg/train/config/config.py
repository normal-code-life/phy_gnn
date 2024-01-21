import abc
import yaml


class BaseConfig(abc.ABC):
    """
    BaseConfig class is an abstract base class defining the basic structure for configuration classes.

    Attributes:
        config: Attribute to store configuration information, loaded using the yaml.unsafe_load method.
    """

    def __init__(self, config: str):
        """
        Constructor to initialize a BaseConfig object.

        Args:
            config (str): String containing configuration information.

        Notes:
            1. Use the yaml.unsafe_load method to load configuration information.
        """
        self.config = yaml.unsafe_load(config)

    def get_config(self):
        """
        Method to retrieve configuration information.

        Returns:
            dict: Dictionary object containing configuration information.
        """
        return self.config
