import numpy as np
from typing import Dict

from torch.utils.data import Dataset

from pkg.utils.logging import init_logger

logger = init_logger("BASE_DATASET")


class BaseDataset(Dataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        logger.info(f"====== init {data_type} data config ======")
        logger.info(data_config)

        self.gpu = data_config["gpu"]

    def __len__(self):
        raise NotImplementedError("please implement __len__ func")

    def get_head_inputs(self, batch_size) -> Dict:

        inputs, _ = self.__getitem__(np.arange(0, batch_size))

        return {key: data for key, data in inputs.items()}
