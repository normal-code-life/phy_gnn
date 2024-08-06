from typing import Dict
import torch
import numpy as np
from torchvision import transforms

from pkg.train.module.data_transform import (CovertToModelInputs, MaxMinNorm,
                                             NormalNorm, SqueezeDataDim,
                                             TFRecordToTensor)
from task.passive_biv.data.datasets import PassiveBiVDataset


class PassiveBiVTrainDataset(PassiveBiVDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)

        self.data_size = np.load(self.data_size_path).astype(np.int64).item()

        # displacement_stats_loaded = np.load(self.displacement_stats_path)
        # self.displacement_stats = {
        #     name: torch.tensor(displacement_stats_loaded[name]) for name in displacement_stats_loaded
        # }

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

