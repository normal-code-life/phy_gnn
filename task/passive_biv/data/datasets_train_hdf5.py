from typing import Dict

import numpy as np
import torch
from torchvision import transforms

from common.constant import MODEL_TRAIN, MAX_VAL, MIN_VAL
from pkg.train.datasets.base_datasets_train import MultiHDF5Dataset
from pkg.train.module.data_transform import CovertToModelInputs, MaxMinNorm, SqueezeDataDim, ToTensor, ClampTensor
from task.passive_biv.data.datasets import FEHeartSageDataset


class FEHeartSageTrainDataset(MultiHDF5Dataset, FEHeartSageDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type, MODEL_TRAIN)

        self.data_size = np.load(self.data_size_path).astype(np.int64).item()

        self._init_transform()

    # init transform data
    def _init_transform(self):
        transform_list = []

        hdf5_to_tensor_config = {
            "context_description": self.context_description,
            "feature_description": self.feature_description,
        }
        transform_list.append(ToTensor(hdf5_to_tensor_config))

        norm_config = {
            "node_coord": self.node_coord_stats_path,
            "fiber_and_sheet": self.fiber_and_sheet_stats_path,
            "shape_coeffs": self.shape_coeff_stats_path,
            "mat_param": self.mat_param_stats_path,
            "pressure": self.pressure_stats_path,
        }

        transform_list.append(MaxMinNorm(norm_config, True, True))

        clamp_config = {
            "stress": {
                MAX_VAL: 20720,
                MIN_VAL: 0,
            }
        }

        transform_list.append((ClampTensor(clamp_config)))

        norm_config_disp = {
            "displacement": self.displacement_stats_path,
        }
        transform_list.append(MaxMinNorm(norm_config_disp))

        norm_config_stress = {
            "stress": {
                MAX_VAL: 20720,
                MIN_VAL: 0,
            }
        }
        transform_list.append(MaxMinNorm(norm_config_stress, setup_val=True))

        # convert data dim
        convert_data_dim_config = {"mat_param": -1, "pressure": -1, "shape_coeffs": -1}
        transform_list.append(SqueezeDataDim(convert_data_dim_config))

        # convert to model inputs
        convert_model_input_config = {"labels": self.labels}

        transform_list.append(CovertToModelInputs(convert_model_input_config, True))

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.data_size

    def get_head_inputs(self, batch_size) -> Dict:
        res: Dict = {}
        for i in range(batch_size):
            inputs, _ = next(self.__iter__())

            inputs = {key: inputs[key].unsqueeze(0) for key in inputs}

            for key in inputs:
                res[key] = torch.concat([res[key], inputs[key]], dim=0) if key in res else inputs[key]

        # for version `fe_heart_sage_v4` we need to input extra `selected_node` and `selected_node_num`:
        # considering it is only for demo, we will use a dummy node num
        _, node_num, _ = res["edges_indices"].shape

        selected_node_num = 300

        selected_node = (
            torch.randint(0, node_num, size=(selected_node_num,), dtype=torch.int64, device="cpu")
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        res["selected_node"] = selected_node

        return res
