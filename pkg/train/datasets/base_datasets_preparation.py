import abc
import os
from typing import Dict

from common.constant import TRAIN_NAME
from pkg.train.datasets.base_datasets import BaseAbstractDataPreparationDataset, BaseAbstractDataset
from pkg.utils.io import check_and_clean_path


class AbstractDataPreparationDataset(BaseAbstractDataPreparationDataset, BaseAbstractDataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        super().__init__(data_config, data_type, *args, **kwargs)

        self.overwrite_data = data_config.get("overwrite_data", False)

        if not check_and_clean_path(self.dataset_path, self.overwrite_data):
            raise ValueError(f"please check your data path and config, some conflict exist {self.dataset_path}")

        if not os.path.exists(self.stats_data_path):
            os.makedirs(self.stats_data_path)

    @abc.abstractmethod
    def prepare_dataset_process(self):
        raise NotImplementedError("Subclasses must implement the prepare_dataset_process method.")

    @abc.abstractmethod
    def _data_generation(self):
        raise NotImplementedError("Subclasses must implement the _data_generation method.")

    @abc.abstractmethod
    def _data_stats(self):
        raise NotImplementedError("Subclasses must implement the _data_stats method.")
