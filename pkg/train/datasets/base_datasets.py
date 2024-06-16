import abc
from typing import Dict

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from pkg.utils.logging import init_logger

logger = init_logger("BASE_DATASET")


class BaseAbstractDataset(abc.ABC):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        logger.info(data_config)

        self.gpu = data_config["gpu"]

        self.data_type = data_type


class BaseDataset(BaseAbstractDataset, Dataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        logger.info(f"====== init {data_type} data config ======")
        super().__init__(data_config, data_type, args, kwargs)

    def __len__(self):
        raise NotImplementedError("please implement __len__ func")

    def get_head_inputs(self, batch_size) -> Dict:
        inputs, _ = self.__getitem__(np.arange(0, batch_size))

        return {key: data for key, data in inputs.items()}


class BaseIterableDataset(BaseAbstractDataset, IterableDataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        logger.info(f"====== init {data_type} data config ======")
        super().__init__(data_config, data_type, args, kwargs)

    def get_head_inputs(self, batch_size) -> Dict:
        inputs, _ = self.__iter__()

        return {key: data for key, data in inputs.items()}

    def __iter__(self):
        raise NotImplementedError("please implement __iter__ func")
