import os
import sys
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numba.typed import List as Numba_List
from torch import Tensor, nn
from torchvision import transforms

from common.constant import DARWIN, MAX_VAL, MIN_VAL
from pkg.data_utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.train.module.data_transform import CovertToModelInputs, MaxMinNorm, ToTensor, UnSqueezeDataDim
from pkg.train.trainer.base_trainer import TrainerConfig
from pkg.utils import io
from pkg.utils.logs import init_logger
from pkg.utils.model_summary import summary_model
from task.passive_biv.data.datasets import FEHeartSageDataset

logger = init_logger("SINGLE_CASE_EVAL")


class FEHeartSageV2Evaluation(FEHeartSageDataset):
    """Evaluation class for FE Heart SAGE model that handles single graph cases.

    This class extends FEHeartSageDataset to provide evaluation functionality for individual test cases.
    It handles data loading, preprocessing, model inference and result saving for single graph evaluations.
    The class supports configurable data transforms and model loading from checkpoints.
    """

    def __init__(self, data_config: Dict, data_type: str, idx: int = 1) -> None:
        """Initialize the evaluation class.

        Args:
            data_config (Dict): Configuration dictionary containing data parameters
            data_type (str): Type of data being used (e.g. 'train', 'eval')
            idx (int, optional): Index of the test case. Defaults to 1.
        """
        super().__init__(data_config, data_type)

        # data preparation param
        # === test case number
        self.idx = idx
        self.device = "cuda" if data_config["gpu"] else "cpu"

        # === param random select edges based on node relative distance
        self.sections = data_config["sections"]
        self.nodes_per_sections = data_config["nodes_per_sections"]

        # data preprocess
        self._init_transform()

        # output path
        self.output_path = f"./output_{self.idx + 1:04d}.csv"

    def single_graph_evaluation(self):
        """Evaluate a single graph case and save results.

        Generates data, applies transforms, runs model inference and saves output to CSV.
        """
        data = self._data_generation()

        transform = self._init_transform()

        inputs, _ = transform(data)

        model = self._load_model()

        logger.info("=== Print Model Structure ===")
        logger.info(model)

        str_summary = summary_model(
            model,
            inputs,
            show_input=True,
            show_hierarchical=True,
            # print_summary=model_summary["print_summary"],
            max_depth=999,
            show_parent_layers=True,
        )

        logger.info(str_summary)

        with torch.no_grad():
            output = model(inputs)

            stats = np.load(self.displacement_stats_path)

            max_val = torch.tensor(stats[MAX_VAL], device=self.device)
            min_val = torch.tensor(stats[MIN_VAL], device=self.device)

            output = output["displacement"].squeeze(0) * (max_val - min_val) + min_val

            df = pd.DataFrame(output.to(self.device).squeeze(0).numpy())
            df.to_csv(self.output_path, index=False)

    def _data_generation(self) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        """Generate input and output data for a single test case.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Tuple containing:
                - context_example: Dictionary with index and points information
                - feature_example: Dictionary with node features and labels
        """
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
        """Generate edges based on node distances.

        Args:
            node_coords: Node coordinates array

        Returns:
            np.ndarray: Generated edges array
        """
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

    def _init_transform(self):
        """Initialize data transformation pipeline.

        Returns:
            transforms.Compose: Composed transformation pipeline
        """
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
        transform_list.append(MaxMinNorm(norm_config))

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

        convert_model_input_config = {"labels": self.labels}

        transform_list.append(CovertToModelInputsWithSelectedNode(convert_model_input_config, True))

        self.transform = transforms.Compose(transform_list)

        return transforms.Compose(transform_list)

    def _load_model(self) -> nn.Module:
        """Load and prepare model for evaluation.

        Returns:
            nn.Module: Loaded PyTorch model in evaluation mode
        """
        model = torch.load(
            f"{self.base_repo_path}/log/{self.task_name}/{self.exp_name}/model/model.pth",
            map_location=torch.device(self.device),
        )

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if hasattr(model, "device"):
            model.device = self.device

        model.eval()

        return model

    @staticmethod
    def total_params_count(model: nn.Module) -> None:
        logger.info(f"print model arch: {model}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'=' * 50}")
        print(f"Model Architecture:")
        print(model)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"{'=' * 50}\n")


class CovertToModelInputsWithSelectedNode(CovertToModelInputs):
    """Convert inputs with selected nodes for model processing."""

    def __init__(self, config: Dict, multi_obj: bool = False, selected_node_num: int = 300) -> None:
        """Initialize the converter.

        Args:
            config (Dict): Configuration dictionary
            multi_obj (bool, optional): Whether using multiple objectives. Defaults to False.
            selected_node_num (int, optional): Number of nodes to select. Defaults to 300.
        """
        super().__init__(config, multi_obj)
        self.selected_node_num = selected_node_num

    def __call__(
        self, sample: Tuple[Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Tensor], Union[Tensor, Dict[str, Tensor]]]:
        """Convert input sample to model format with selected nodes.

        Args:
            sample: Tuple of input and label dictionaries

        Returns:
            Tuple containing processed inputs and labels
        """
        inputs, labels = super().__call__(sample)

        batch_size, node_num, _ = inputs["edges_indices"].shape

        selected_node = torch.arange(node_num, device="cpu").unsqueeze(0).expand(batch_size, -1)

        inputs["selected_node"] = selected_node

        return inputs, labels


if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])
    task_dir = io.get_repo_path(cur_path)
    sys.argv.extend(
        [
            "--repo_path",
            f"{task_dir}",
            "--task_name",
            "passive_biv",
            "--model_name",
            "fe_heart_sage_v4",
            "--config_name",
            "train_config",
            "--task_type",
            "model_train",
        ]
    )

    config = TrainerConfig()

    # by default, we use gpu to do the single case test
    config.task_data["gpu"] = False
    config.task_data["sections"] = [0, 20, 100, 250, 500, 1000]
    config.task_data["nodes_per_sections"] = [20, 30, 30, 10, 10]

    evaluation = FEHeartSageV2Evaluation(config.task_data, "eval", 3)

    evaluation.single_graph_evaluation()
