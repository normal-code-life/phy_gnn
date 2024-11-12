from typing import Dict

import numpy as np
import pandas as pd
import torch
from numba.typed import List as Numba_List
from torch import nn
from torchvision import transforms

from common.constant import DARWIN, MAX_VAL, MIN_VAL
from pkg.data_utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.train.module.data_transform import CovertToModelInputs, MaxMinNorm, ToTensor, UnSqueezeDataDim
from pkg.utils.logs import init_logger
from task.passive_biv.data.datasets import FEHeartSageV2Dataset, import_data_config

logger = init_logger("single_case_eval")


class FEHeartSageV2Evaluation(FEHeartSageV2Dataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, data_config: Dict, data_type: str, idx: int = 1) -> None:
        super().__init__(data_config, data_type)

        # data preparation param
        # === test case number
        self.idx = idx

        # === param random select edges based on node relative distance
        self.sections = data_config["sections"]
        self.nodes_per_sections = data_config["nodes_per_sections"]

        # data preprocess
        self._init_transform()

        # output path
        self.output_path = f"./output_{self.idx + 1:04d}.csv"

    def single_graph_evaluation(self):
        data = self._data_generation()

        transform = self._init_transform()

        inputs, _ = transform(data)

        model = self._load_model()

        with torch.no_grad():
            output = model(inputs)

            stats = np.load(self.displacement_stats_path)

            max_val = torch.tensor(stats[MAX_VAL])
            min_val = torch.tensor(stats[MIN_VAL])

            output = output.squeeze(0) * (max_val - min_val) + min_val

            df = pd.DataFrame(output.squeeze(0).numpy())
            df.to_csv(self.output_path, index=False)

    def _data_generation(self) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        # read global features
        data_global_feature = np.loadtxt(self.global_feature_data_path, delimiter=",")
        data_shape_coeff = np.loadtxt(self.shape_data_path, delimiter=",")

        read_file_name = f"/ct_case_{self.idx + 1:04d}.csv"  # e.g. ct_case_0005

        record_inputs = np.loadtxt(self.inputs_data_path + read_file_name, delimiter=",", dtype=np.float32)

        record_outputs = np.loadtxt(self.outputs_data_path + read_file_name, delimiter=",", dtype=np.float32)

        points = record_inputs.shape[0]

        edge: np.ndarray = self._generate_distance_based_edges(record_inputs[:, 0:3])

        context_example = {
            "index": np.array([np.int32(self.idx)]),
            "points": np.array([np.int32(points)]),
        }

        feature_example = {
            "node_coord": record_inputs[:, 0:3],
            "laplace_coord": record_inputs[:, 3:11],
            "fiber_and_sheet": record_inputs[:, 11:17],
            "edges_indices": edge[0].astype(np.int64),
            "mat_param": data_global_feature[:, 1:7][self.idx],
            "pressure": data_global_feature[:, 7:9][self.idx],
            "shape_coeffs": data_shape_coeff[:, 1:60][self.idx],
            "displacement": record_outputs[:, 0:3],
            "stress": record_outputs[:, 3:4],
        }

        return context_example, feature_example

    def _generate_distance_based_edges(self, node_coords) -> np.ndarray:
        if self.platform == DARWIN:
            return generate_distance_based_edges_ny(
                node_coords[np.newaxis, :, :], [0], self.sections, self.nodes_per_sections
            )

        sections = self.sections
        nodes_per_sections = self.nodes_per_sections

        sections_nb = Numba_List()
        [sections_nb.append(x) for x in sections]

        nodes_per_section_nb = Numba_List()
        [nodes_per_section_nb.append(x) for x in nodes_per_sections]

        # need to expand the axis and align with the other method
        return generate_distance_based_edges_nb(node_coords, sections_nb, nodes_per_section_nb)[np.newaxis, :].astype(
            np.int32
        )

    # init transform data
    def _init_transform(self):
        transform_list = []

        hdf5_to_tensor_config = {
            "context_description": self.context_description,
            "feature_description": self.feature_description,
        }
        transform_list.append(ToTensor(hdf5_to_tensor_config))

        max_min_norm_config = {
            "node_coord": self.node_coord_stats_path,
            "fiber_and_sheet": self.fiber_and_sheet_stats_path,
            "shape_coeffs": self.shape_coeff_stats_path,
            "mat_param": self.mat_param_stats_path,
            "pressure": self.pressure_stats_path,
        }

        transform_list.append(MaxMinNorm(max_min_norm_config, True, True))

        norm_config = {
            "displacement": self.displacement_stats_path,
            "stress": self.stress_stats_path,
        }
        transform_list.append(MaxMinNorm(norm_config, True))

        unsqueeze_data_dim_config = {
            "node_coord": 0,
            "laplace_coord": 0,
            "fiber_and_sheet": 0,
            "edges_indices": 0,
            "displacement": 0,
            "stress": 0,
            "mat_param": 0,
            "pressure": 0,
            "shape_coeffs": 0,
        }
        transform_list.append(UnSqueezeDataDim(unsqueeze_data_dim_config))

        # convert to model inputs
        convert_model_input_config = {"labels": self.labels}

        transform_list.append(CovertToModelInputs(convert_model_input_config, True))

        return transforms.Compose(transform_list)

    def _load_model(self) -> nn.Module:
        model = torch.load(
            f"{self.base_repo_path}/log/{self.task_name}/{self.exp_name}/model/model.pth",
            map_location=torch.device("cpu"),
        )

        model.eval()

        return model


if __name__ == "__main__":
    config = import_data_config()

    evaluation = FEHeartSageV2Evaluation(config, "eval", 2)

    evaluation.single_graph_evaluation()
