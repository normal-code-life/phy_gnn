import abc


class BaseConfig(abc.ABC):
    """
    BaseConfig class is an abstract base class defining the basic structure for configuration classes.

    Attributes:
        config: Attribute to store configuration information, loaded using the yaml.unsafe_load method.
    """

    def get_config(self):
        raise NotImplementedError("please implement 'get_config' method")
