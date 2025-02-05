from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms

from common.constant import MAX_VAL, MIN_VAL, MODEL_TRAIN, PERC_10_VAL, PERC_90_VAL
from pkg.train.datasets.base_datasets_train import MultiHDF5Dataset
from pkg.train.module.data_transform import ClampTensor, CovertToModelInputs, MaxMinNorm, SqueezeDataDim, ToTensor
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

        climp_config = {
            "displacement": {
                MAX_VAL: 2.688125,
                MIN_VAL: -2.8395823,
            }
        }

        transform_list.append(ClampTensor(climp_config))

        norm_config = {
            "node_coord": self.node_coord_stats_path,
            "fiber_and_sheet": self.fiber_and_sheet_stats_path,
            "shape_coeffs": self.shape_coeff_stats_path,
            "mat_param": self.mat_param_stats_path,
            "pressure": self.pressure_stats_path,
        }

        transform_list.append(MaxMinNorm(norm_config, True, True))

        norm_config = {
            "displacement": self.displacement_stats_path,
            "stress": self.stress_stats_path,
            "replace_by_perc": {
                MIN_VAL: PERC_10_VAL,
                MAX_VAL: PERC_90_VAL,
            },
        }
        transform_list.append(MaxMinNorm(norm_config, True))

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


class CovertToModelInputsRandom(CovertToModelInputs):
    def __init__(self, config: Dict, multi_obj: bool = False, selected_node_num: int = 300) -> None:
        super().__init__(config, multi_obj)
        self.selected_node_num = selected_node_num

    def __call__(
        self, sample: Tuple[Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Tensor], Union[Tensor, Dict[str, Tensor]]]:
        inputs, labels = super().__call__(sample)

        node_num, _ = inputs["edges_indices"].shape

        selected_node = torch.randint(0, node_num, size=(self.selected_node_num,), dtype=torch.int64)

        inputs["selected_node"] = selected_node
        labels["selected_node"] = selected_node

        return inputs, labels
