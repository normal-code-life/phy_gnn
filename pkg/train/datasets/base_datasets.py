import abc
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from pkg.utils.logging import init_logger

logger = init_logger("BASE_DATASET")


class BaseAbstractDataset(abc.ABC):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        logger.info(data_config)

        self.gpu = data_config["gpu"]
        self.cuda_core = data_config.get("cuda_core", "gpu:0")

        self.data_type = data_type

    def __len__(self):
        raise NotImplementedError("please implement __len__ func")


class BaseDataset(BaseAbstractDataset, Dataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        logger.info(f"====== init {data_type} data config ======")
        super().__init__(data_config, data_type, args, kwargs)

    def get_head_inputs(self, batch_size) -> Dict:
        inputs, _ = self.__getitem__(np.arange(0, batch_size))

        return {key: data for key, data in inputs.items()}


class BaseIterableDataset(BaseAbstractDataset, IterableDataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        logger.info(f"====== init {data_type} data config ======")
        super().__init__(data_config, data_type, args, kwargs)

    def get_head_inputs(self, batch_size) -> Dict:
        res = {}
        for i in range(batch_size):
            inputs, _ = next(self.__iter__())

            inputs = {key: inputs[key].unsqueeze(0) for key in inputs}

            for key in inputs:
                res[key] = torch.concat([res[key], inputs[key]], dim=0) if key in res else inputs[key]

        return res

    def __iter__(self):
        raise NotImplementedError("please implement __iter__ func")
