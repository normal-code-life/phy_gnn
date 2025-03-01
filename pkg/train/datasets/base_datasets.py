import abc
import os
import platform
from typing import Dict, Optional

from pkg.train.datasets import logger


class BaseAbstractDataset(abc.ABC):
    """Base abstract class for dataset handling.

    Provides core functionality for dataset preparation and training, including path setup
    and hardware configuration. Subclasses implement specific data processing logic.

    Attributes:
        base_data_path (str): Base data directory path
        base_task_path (str): Base task directory path
        gpu (bool): Whether to use GPU
        cuda_core (str): CUDA core identifier
        data_type (str): Dataset type (e.g. train, test)
        exp_name (str, optional): Experiment name
    """

    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        """Initialize dataset with config.

        Args:
            data_config (Dict): Configuration for paths and hardware
            data_type (str): Dataset type (e.g. train, test)
            *args: Additional args
            **kwargs: Additional kwargs
        """
        logger.info(f"=== Init BaseAbstractDataset {data_type} data config start ===")
        logger.info(f"data_config is: {data_config}")

        # common config
        # === Hardware configuration
        self.gpu = data_config.get("gpu", False)
        self.cuda_core = data_config.get("cuda_core", 0)
        self.platform = platform.system()

        # === model_name
        self.model_name = data_config["model_name"]

        # === data type
        self.task_name = data_config["task_name"]
        self.data_type = data_type

        # === exp
        self.exp_name = data_config.get("exp_name", "")

        # common path
        # === base path
        self.base_repo_path = data_config["repo_path"]
        self.base_data_path = data_config["task_data_path"]
        self.base_task_path = data_config["task_path"]

        if not os.path.isdir(self.base_data_path):
            raise NotADirectoryError(f"No directory at: {self.base_data_path}")

        # === traditional model training dataset path (non-tfrecord version)
        self.stats_data_path = f"{self.base_data_path}/stats"
        self.dataset_path = f"{self.base_data_path}/datasets/{self.data_type}"
        self.data_size_path = f"{self.stats_data_path}/{self.data_type}_data_size.npy"

        logger.info(f"base_data_path is {self.base_data_path}")
        logger.info(f"base_task_path is {self.base_task_path}")
        logger.info(f"stats_data_path is {self.stats_data_path}")
        logger.info(f"dataset_path is {self.dataset_path}")
        logger.info(f"data_size_path is {self.data_size_path}")

        # hdf5 config
        self.dataset_h5_path = f"{self.dataset_path}" + "/data_{}.h5"

        logger.info(f"dataset_h5_path is {self.dataset_h5_path}")

        # tfrecord config
        # === tfrecord model training dataset path (tfrecord version)
        self.tfrecord_data_path = f"{self.dataset_path}" + "/data_{}.tfrecord"

        logger.info(f"tfrecord_data_path is {self.tfrecord_data_path}")

        # === tfrecord file compression type
        self.compression_type = None

        # tfrecord features
        self.context_description: Optional[Dict[str, str]] = None  # please overwrite this variable

        self.feature_description: Optional[Dict[str, str]] = None  # please overwrite this variable

        logger.info(f"=== Init BaseAbstractDataset {data_type} data config done ===")


class BaseAbstractDataPreparationDataset(abc.ABC):
    """Base abstract class for dataset preparation.

    Defines interface for data generation and statistics computation.
    Subclasses implement specific preparation logic.

    Methods:
        prepare_dataset_process(): Main dataset preparation workflow
        _data_generation(): Generate dataset
        _data_stats(): Compute dataset statistics
    """

    @abc.abstractmethod
    def prepare_dataset_process(self):
        """Execute dataset preparation workflow.

        Subclasses implement specific preparation steps.
        """
        raise NotImplementedError("Subclasses must implement the prepare_dataset_process method.")

    @abc.abstractmethod
    def _data_generation(self):
        """Generate dataset.

        Subclasses implement data generation logic.
        """
        raise NotImplementedError("Subclasses must implement the _data_generation method.")

    @abc.abstractmethod
    def _data_stats(self):
        """Compute dataset statistics.

        Subclasses implement statistics computation.
        """
        raise NotImplementedError("Subclasses must implement the _data_stats method.")


class BaseAbstractTrainDataset(BaseAbstractDataset):
    @abc.abstractmethod
    def get_head_inputs(self, batch_size: int) -> Dict:
        """Get model head inputs for architecture visualization.

        Args:
            batch_size (int): Number of samples to generate

        Returns:
            Dict: Model head inputs
        """
        raise NotImplementedError("Subclasses must implement get_head_inputs method")

    @abc.abstractmethod
    def __len__(self):
        """Get dataset size.

        Returns:
            int: Number of samples in dataset
        """
        raise NotImplementedError("Subclasses must implement __len__ method")
