import abc
from typing import Dict

from common.constant import TRAIN_NAME
from pkg.train.datasets import logger
from pkg.train.datasets.base_datasets import BaseAbstractDataPreparationDataset, BaseAbstractDataset
from pkg.utils.io import check_and_clean_path


class AbstractDataPreparationDataset(BaseAbstractDataPreparationDataset, BaseAbstractDataset):
    """Abstract base class for dataset preparation.

    This class handles the preparation of datasets including data generation, statistics calculation,
    and size tracking. It provides a framework for implementing dataset-specific preparation logic.

    Args:
        data_config (Dict): Configuration dictionary containing dataset parameters
        data_type (str): Type of dataset (e.g. 'train', 'validation')
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments

    Attributes:
        overwrite_data (bool): Whether to overwrite existing dataset files
        overwrite_stats (bool): Whether to overwrite existing statistics files
    """

    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        super().__init__(data_config, data_type, *args, **kwargs)

        self.overwrite_data = data_config.get("overwrite_data", False)
        self.overwrite_stats = data_config.get("overwrite_stats", False)

    def prepare_dataset_process(self) -> None:
        """Main dataset preparation process.

        Handles the full dataset preparation workflow:
        1. Generates dataset if needed or requested via overwrite
        2. Calculates statistics for training data
        3. Records total dataset size
        """
        logger.info("=== Starting dataset preparation process ===")

        if check_and_clean_path(self.dataset_path, self.overwrite_data):
            logger.info(f"Generating dataset at: {self.dataset_path}")
            self._data_generation()
        else:
            logger.info(f"Dataset already exists at: {self.dataset_path}")

        if self.data_type == TRAIN_NAME and check_and_clean_path(self.stats_data_path, self.overwrite_stats):
            self._data_stats()

        self._data_stats_total_size()

        logger.info("=== Dataset preparation process complete ===")

    @abc.abstractmethod
    def _data_generation(self) -> None:
        """Generate the dataset.

        Must be implemented by subclasses to handle dataset-specific generation logic.
        """
        raise NotImplementedError("Subclasses must implement the _data_generation method.")

    @abc.abstractmethod
    def _data_stats(self) -> None:
        """Calculate dataset statistics.

        Must be implemented by subclasses to compute relevant dataset statistics.
        """
        raise NotImplementedError("Subclasses must implement the _data_stats method.")

    @abc.abstractmethod
    def _data_stats_total_size(self) -> None:
        """Calculate and record total dataset size.

        Must be implemented by subclasses to determine the total size of the dataset.
        """
        raise NotImplementedError("Subclasses must implement the _data_stats_total_size method.")
