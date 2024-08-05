import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tfrecord
import torch
import torch.utils.data
from torchvision import transforms

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.train.module.data_transform import (CovertToModelInputs, MaxMinNorm,
                                             NormalNorm, SqueezeDataDim,
                                             TFRecordToTensor)
from task.passive_biv.data.datasets import PassiveBiVDataset


class PassiveBiVTrainDataset(PassiveBiVDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)
        # labels
        self.data_size = np.load(self.data_size_path).astype(np.int64).item()

        # file
        self.num_of_files = len(os.listdir(self.tfrecord_path))

        # init tfrecord loader config
        self.shuffle_queue_size = data_config.get("shuffle_queue_size", 5)

        self.transform: Optional[transforms.Compose] = None

        self._init_transform()

    # init transform data
    def _init_transform(self):
        transform_list = []

        # feature normalization
        tfrecord_to_tensor_config = {
            "context_description": self.context_description,
            "feature_description": self.feature_description,
        }
        transform_list.append(TFRecordToTensor(tfrecord_to_tensor_config))

        max_min_norm_config = {
            "node_coord": self.node_coord_stats_path,
            "fiber_and_sheet": self.fiber_and_sheet_stats_path,
            "shape_coeffs": self.shape_coeff_stats_path,
            "mat_param": self.mat_param_stats_path,
            "pressure": self.pressure_stats_path,
        }

        transform_list.append(MaxMinNorm(max_min_norm_config, False))

        if self.data_type == TRAIN_NAME:
            normal_norm_config = {
                "displacement": self.displacement_stats_path,
                "stress": self.stress_stats_path,
            }
            transform_list.append(NormalNorm(normal_norm_config))

        # convert data dim
        convert_data_dim_config = {"mat_param": -1, "pressure": -1, "shape_coeffs": -1}
        transform_list.append(SqueezeDataDim(convert_data_dim_config))

        # convert to model inputs
        convert_model_input_config = {"labels": self.labels}

        transform_list.append(CovertToModelInputs(convert_model_input_config, True))

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.data_size

    def __iter__(self) -> (Dict, torch.Tensor):
        shift, num_workers = 0, 0

        worker_info = torch.utils.data.get_worker_info()
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
