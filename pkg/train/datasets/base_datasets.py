import abc
from typing import Dict, Optional, Set
import tfrecord
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import os
from pkg.utils.logging import init_logger
from torchvision import transforms

logger = init_logger("BASE_DATASET")


class BaseAbstractDataset(abc.ABC):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        logger.info(data_config)

        # path
        # === base path
        self.base_data_path = f"{data_config['task_data_path']}"
        self.base_task_path = f"{data_config['task_path']}"

        # other config
        # === cpu/gpu
        self.gpu = data_config["gpu"]
        self.cuda_core = data_config.get("cuda_core", "gpu:0")

        # === data type
        self.data_type = data_type

        # === exp
        self.exp_name = data_config.get("exp_name", None)

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

            inputs = {key: inputs[key].cuda().unsqueeze(0) if self.gpu else inputs[key].unsqueeze(0) for key in inputs}

            for key in inputs:
                res[key] = torch.concat([res[key], inputs[key]], dim=0) if key in res else inputs[key]

        return res

    def __iter__(self):
        raise NotImplementedError("please implement __iter__ func")


class MultiTFRecordDataset(BaseIterableDataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        super().__init__(data_config, data_type, args, kwargs)

        # path
        self.stats_data_path = f"{self.base_data_path}/stats"
        self.tfrecord_path = f"{self.base_data_path}/tfrecord/{self.data_type}"
        self.tfrecord_data_path = f"{self.tfrecord_path}" + "/data_{}.tfrecord"

        logger.info(f"base_data_path is {self.base_data_path}")
        logger.info(f"base_task_path is {self.base_task_path}")
        logger.info(f"stats_data_path is {self.stats_data_path}")
        logger.info(f"tfrecord_path is {self.tfrecord_path}")
        logger.info(f"tfrecord_data_path is {self.tfrecord_data_path}")

        if not os.path.exists(self.stats_data_path):
            os.makedirs(self.stats_data_path)

        if not os.path.exists(self.tfrecord_path):
            os.makedirs(self.tfrecord_path)

        # config
        # === path file size
        self.num_of_files = len(os.listdir(self.tfrecord_path))

        # === file compression
        self.compression_type = None

        # === shuffle queue size
        self.shuffle_queue_size = data_config.get("shuffle_queue_size", 5)

        # features
        self.context_description: Optional[Dict[str, str]] = None  # please overwrite this variable

        self.feature_description: Optional[Dict[str, str]] = None  # please overwrite this variable

        self.labels: Optional[Set[str]] = None

        # transform
        self.transform: Optional[transforms.Compose] = None

    def _init_transform(self):
        return

    def __iter__(self) -> (Dict, torch.Tensor):
        shift, num_workers = 0, 0

        worker_info = get_worker_info()
        if worker_info is not None:
            shift, num_workers = worker_info.id, worker_info.num_workers

        if num_workers > self.num_of_files:
            raise ValueError("the num of workers should be small or equal to num of files")

        if num_workers == 0:
            splits = {str(num): 1.0 for num in range(self.num_of_files)}
        else:
            splits = {str(num): 1.0 for num in range(self.num_of_files) if num % num_workers == shift}

        it = tfrecord.multi_tfrecord_loader(
            data_pattern=self.tfrecord_data_path,
            index_pattern=None,
            splits=splits,
            description=self.context_description,
            sequence_description=self.feature_description,
            compression_type=self.compression_type,
            infinite=False,
        )

        if self.shuffle_queue_size:
            it = tfrecord.shuffle_iterator(it, self.shuffle_queue_size)  # noqa

        it = map(self.transform, it)

        return it
