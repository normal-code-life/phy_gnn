import abc
import os
from typing import Dict, Optional

import numpy as np
import tfrecord
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision import transforms

from common.constant import TRAIN_NAME
from pkg.train.datasets.base_datasets import BaseAbstractDataset, BaseAbstractTrainDataset
from pkg.train.datasets.reader_hdf5 import multi_hdf5_loader, shuffle_iterator


class AbstractTrainDataset(BaseAbstractTrainDataset, BaseAbstractDataset, Dataset):
    @abc.abstractmethod
    def get_head_inputs(self, batch_size: int) -> Dict:
        raise NotImplementedError("Subclasses must implement get_head_inputs method")

    def __len__(self):
        return np.load(self.data_size_path).astype(np.int64)


class BaseDataset(AbstractTrainDataset):
    def get_head_inputs(self, batch_size) -> Dict:
        inputs, _ = self.__getitem__(np.arange(0, batch_size))

        return {key: data.cpu() for key, data in inputs.items()}


class BaseIterableDataset(AbstractTrainDataset, IterableDataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        super().__init__(data_config, data_type, *args, **kwargs)

        # overwrite the stats data path since during training,
        # we could only leverage on the train data stats except the data size
        self.stats_data_path = f"{self.base_data_path}/stats/{TRAIN_NAME}"

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


class MultiTFRecordDataset(BaseIterableDataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        super().__init__(data_config, data_type, *args, **kwargs)
        # config
        # === path file size
        self.num_of_files = len(os.listdir(self.dataset_path))

        # === file compression
        self.compression_type = None

        # === shuffle queue size
        self.shuffle_queue_size = data_config.get("shuffle_queue_size", 5)

        # transform
        self.transform: Optional[transforms.Compose] = None

    def _init_transform(self):
        return

    def __iter__(self) -> (Dict, torch.Tensor):
        shift, num_workers = 0, 0

        worker_info = get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
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


class MultiHDF5Dataset(BaseIterableDataset):
    def __init__(self, data_config: Dict, data_type: str, *args, **kwargs) -> None:
        super().__init__(data_config, data_type, *args, **kwargs)
        # config
        # === path file size
        self.num_of_files = len(os.listdir(self.dataset_path))

        # === file compression
        self.compression_type = None

        # === shuffle queue size
        self.shuffle_queue_size = data_config.get("shuffle_queue_size", 5)

        # transform
        self.transform: Optional[transforms.Compose] = None

    def _init_transform(self):
        return

    def __iter__(self) -> (Dict, torch.Tensor):
        shift, num_workers = 0, 0

        worker_info = get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
            shift, num_workers = worker_info.id, worker_info.num_workers

        if num_workers > self.num_of_files:
            raise ValueError("the num of workers should be small or equal to num of files")

        if num_workers == 0:
            splits = {str(num) for num in range(self.num_of_files)}
        else:
            splits = {str(num) for num in range(self.num_of_files) if num % num_workers == shift}

        it = multi_hdf5_loader(
            data_pattern=self.dataset_h5_path,
            splits=splits,
            infinite=False,
            description=self.context_description,
            sequence_description=self.feature_description,
        )

        if self.shuffle_queue_size:
            it = shuffle_iterator(it, self.shuffle_queue_size)  # noqa

        it = map(self.transform, it)

        return it
