import abc
from typing import Dict

from common.constant import TRAIN_NAME
from pkg.train.datasets import logger
from pkg.train.datasets.base_datasets import BaseAbstractDataPreparationDataset, BaseAbstractDataset
from pkg.utils.io import check_and_clean_path


class AbstractDataPreparationDataset(BaseAbstractDataPreparationDataset, BaseAbstractDataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        super().__init__(data_config, data_type, *args, **kwargs)

        self.overwrite_data = data_config.get("overwrite_data", False)
        self.overwrite_stats = data_config.get("overwrite_stats", False)

    def prepare_dataset_process(self):
        if check_and_clean_path(self.dataset_path, self.overwrite_data):
            logger.info(f"data generation starts: {self.dataset_path}")
            self._data_generation()
        else:
            logger.info(f"data already exists, no overwrite: {self.dataset_path}")

        if self.data_type == TRAIN_NAME and check_and_clean_path(self.stats_data_path, self.overwrite_stats):
            self._data_stats()

        self._data_stats_total_size()

    @abc.abstractmethod
    def _data_generation(self):
        raise NotImplementedError("Subclasses must implement the _data_generation method.")

    @abc.abstractmethod
    def _data_stats(self):
        raise NotImplementedError("Subclasses must implement the _data_stats method.")

    @abc.abstractmethod
    def _data_stats_total_size(self):
        raise NotImplementedError("Subclasses must implement the _data_stats_total_size method.")
