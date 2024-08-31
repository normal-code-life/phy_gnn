from typing import Dict, List, Optional

import h5py
import numpy as np
from numba.typed import List as Numba_List

from common.constant import DARWIN, TEST_NAME, TRAIN_NAME
from pkg.data.utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.data.utils.stats import stats_analysis
from pkg.train.datasets.base_datasets_preparation import AbstractDataPreparationDataset
from task.passive_biv.fe_heart_sage_v2.data import logger
from task.passive_biv.fe_heart_sage_v2.data.datasets import FEHeartSageV2Dataset


class PassiveBiVPreparationDataset(AbstractDataPreparationDataset, FEHeartSageV2Dataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super(PassiveBiVPreparationDataset, self).__init__(data_config, data_type)
        # sample indices
        self.sample_indices = data_config["sample_indices"]

        # other parameter
        # === param sample size for each files
        self.chunk_file_size = data_config["chunk_file_size"]

        # === param random select edges based on node relative distance
        self.sections = data_config["sections"]
        self.nodes_per_sections = data_config["nodes_per_sections"]
        self.train_down_sampling_node: Optional[float] = data_config.get("train_down_sampling_node", None)
        self.val_down_sampling_node: Optional[float] = data_config.get("val_down_sampling_node", None)

        logger.info(f"data config info: {data_config}")

        logger.info(f"====== finish {self.__class__.__name__} {data_type} data config ======")

    def _data_generation(self):
        # read global features
        data_global_feature = np.loadtxt(self.global_feature_data_path, delimiter=",")
        data_shape_coeff = np.loadtxt(self.shape_data_path, delimiter=",")

        sample_indices: List[np.ndarray] = np.array_split(
            self.sample_indices, len(self.sample_indices) // self.chunk_file_size
        )

        for i, indices in enumerate(sample_indices):
            datasets = []

            for idx in indices:
                # read sample inputs
                read_file_name = f"/ct_case_{idx + 1:04d}.csv"  # e.g. ct_case_0005

                record_inputs = np.loadtxt(self.inputs_data_path + read_file_name, delimiter=",", dtype=np.float32)

                record_outputs = np.loadtxt(self.outputs_data_path + read_file_name, delimiter=",", dtype=np.float32)

                points = record_inputs.shape[0]

                if (self.train_down_sampling_node or self.val_down_sampling_node) and self.data_type != TEST_NAME:
                    record_inputs, record_outputs = self._down_sampling_node(record_inputs, record_outputs)

                edge: np.ndarray = self._generate_distance_based_edges(record_inputs[:, 0:3])

                datasets.append(
                    {
                        "index": np.array([np.int32(idx)]),
                        "points": np.array([np.int32(points)]),
                        "node_coord": record_inputs[:, 0:3],
                        "laplace_coord": record_inputs[:, 3:11],
                        "fiber_and_sheet": record_inputs[:, 11:17],
                        "edges_indices": edge[0].astype(np.int32),
                        "mat_param": data_global_feature[:, 1:7][idx],
                        "pressure": data_global_feature[:, 7:9][idx],
                        "shape_coeffs": data_shape_coeff[:, 1:59][idx],
                        "displacement": record_outputs[:, 0:3],
                        "stress": record_outputs[:, 3:4],
                    }
                )

            with h5py.File(self.dataset_h5_path.format(i), "w") as f:
                for idx, sample_dict in enumerate(datasets):
                    group: h5py.Group = f.create_group(f"idx_{idx}")
                    for key, value in sample_dict.items():
                        group.create_dataset(key, data=value)

            logger.info(f"data_generation {i}: {indices} done")

    def _down_sampling_node(self, record_inputs: np.ndarray, record_outputs: np.ndarray) -> (np.ndarray, np.ndarray):
        num_nodes, record_inputs_dim = record_inputs.shape

        _, record_outputs_dim = record_outputs.shape

        num_down_sample_node = (
            self.train_down_sampling_node if self.data_type == TRAIN_NAME else self.val_down_sampling_node
        )

        if num_down_sample_node > num_nodes:
            raise ValueError("num_down_sample_node error, please carefully choice the node number")

        select_nodes = np.random.choice(num_nodes, size=num_down_sample_node, replace=False)

        return record_inputs[select_nodes, :], record_outputs[select_nodes, :]

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

    def _data_stats(self) -> None:
        # we only allow train data stats write to path
        write_to_path = True

        self._data_node_stats(write_to_path)

        self._data_global_feature_stats(write_to_path)

        self._data_label_stats(write_to_path)

    def _data_node_stats(self, write_to_path: bool) -> None:
        # fmt: off
        node_coord_set: Optional[np.ndarray] = None
        laplace_coord_set: Optional[np.ndarray] = None
        fiber_and_sheet_set: Optional[np.ndarray] = None

        for idx in range(len(self.sample_indices)):
            read_file_name = f"/ct_case_{idx + 1:04d}.csv"  # e.g. ct_case_0005
            record_inputs = np.loadtxt(self.inputs_data_path + read_file_name, delimiter=",")

            node_coord = record_inputs[:, 0:3]
            node_coord_set = (
                node_coord if node_coord_set is None else np.concatenate([node_coord_set, node_coord], axis=0)
            )

            laplace_coord = record_inputs[:, 3:11]
            laplace_coord_set = (
                laplace_coord if laplace_coord_set is None else np.concatenate([laplace_coord_set, laplace_coord], axis=0)  # noqa
            )

            fiber_and_sheet = record_inputs[:, 11:17]
            fiber_and_sheet_set = (
                fiber_and_sheet if fiber_and_sheet_set is None else np.concatenate([fiber_and_sheet_set, fiber_and_sheet], axis=0)  # noqa
            )

        stats_analysis("node_coord", node_coord_set, 0, self.node_coord_stats_path, logger, write_to_path)  # noqa
        stats_analysis("laplace_coord", laplace_coord_set, 0, self.node_laplace_coord_stats_path, logger, write_to_path)  # noqa
        stats_analysis("fiber_and_sheet", fiber_and_sheet_set, 0, self.fiber_and_sheet_stats_path, logger, write_to_path)  # noqa

        # fmt: on

    def _data_global_feature_stats(self, write_to_path: bool) -> None:
        # fmt: off
        data_global_feature = np.loadtxt(self.global_feature_data_path, delimiter=",")
        data_shape_coeff = np.loadtxt(self.shape_data_path, delimiter=",")

        stats_analysis("mat_param", data_global_feature[:, 1:7], 0, self.mat_param_stats_path, logger, write_to_path)  # noqa
        stats_analysis("pressure", data_global_feature[:, 7:9], 0, self.pressure_stats_path, logger, write_to_path)  # noqa
        stats_analysis("shape_coeffs", data_shape_coeff[:, 1:], 0, self.shape_coeff_stats_path, logger, write_to_path)  # noqa

        # fmt: on

    def _data_label_stats(self, write_to_path: bool) -> None:
        # fmt: off
        displacement_set: Optional[np.ndarray] = None
        stress_set: Optional[np.ndarray] = None

        for idx in range(len(self.sample_indices)):
            read_file_name = f"/ct_case_{idx + 1:04d}.csv"  # e.g. ct_case_0005
            record_output = np.loadtxt(self.outputs_data_path + read_file_name, delimiter=",")

            displacement = record_output[:, 0:3]
            displacement_set = (
                displacement if displacement_set is None else np.concatenate([displacement_set, displacement], axis=0)
            )

            stress = record_output[:, 3: 4]
            stress_set = (
                stress if stress_set is None else np.concatenate([stress_set, stress], axis=0)
            )

        stats_analysis("displacement", displacement_set, 0, self.displacement_stats_path, logger, write_to_path)  # noqa
        stats_analysis("stress", stress_set, 0, self.stress_stats_path, logger, write_to_path)

        # fmt: on

    def _data_stats_total_size(self) -> None:
        np.save(self.data_size_path, self.sample_indices.shape[0])
