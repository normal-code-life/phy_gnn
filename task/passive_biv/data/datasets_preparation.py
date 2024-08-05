import multiprocessing as mp
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import tfrecord

from common.constant import TRAIN_NAME
from pkg.data.utils.edge_generation import generate_distance_based_edges
from pkg.data.utils.stats import stats_analysis
from pkg.utils.io import check_and_clean_path
from task.passive_biv.data.datasets import PassiveBiVDataset, logger


class PassiveBiVPreparationDataset(PassiveBiVDataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super(PassiveBiVPreparationDataset, self).__init__(data_config, data_type)
        # param job related
        self.overwrite_data = data_config.get("overwrite_data", False)
        self.num_processes = data_config.get("num_processes", mp.cpu_count())

        # sample indices
        self.sample_indices = data_config["sample_indices"]

        # other parameter
        # === param sample size for each files
        self.chunk_file_size = data_config["chunk_file_size"]

        # === param random select edges based on node relative distance
        self.sections = data_config["sections"]
        self.nodes_per_section = data_config["nodes_per_section"]

        logger.info(f"data config info: {data_config}")

        logger.info(f"====== finish {self.__class__.__name__} {data_type} data config ======")

    def prepare_dataset(self):
        self._data_generation()

        self._data_stats()

        logger.info(f"====== finish {self.__class__.__name__} {self.data_type} preparation done ======\n")

    def _data_generation(self):
        if not check_and_clean_path(self.tfrecord_path, self.overwrite_data):
            return

        # read global features
        data_global_feature = np.loadtxt(self.global_feature_data_path, delimiter=",")
        data_shape_coeff = np.loadtxt(self.shape_data_path, delimiter=",")

        sample_indices: List[np.ndarray] = np.array_split(
            self.sample_indices, len(self.sample_indices) // self.chunk_file_size
        )

        with mp.Pool(processes=self.num_processes) as pool:
            results = [
                pool.apply_async(self._single_file_write_process, (i, indices, data_global_feature, data_shape_coeff))
                for i, indices in enumerate(sample_indices)
            ]

            pool.close()
            pool.join()

            for result in results:
                logger.info(result.get())

    def _single_file_write_process(
        self, file_i: int, indices: np.ndarray, data_global_feature: np.ndarray, data_shape_coeff: np.ndarray
    ) -> str:
        file_path_group = self.tfrecord_data_path.format(file_i)

        writer = tfrecord.TFRecordWriter(file_path_group)

        for idx in indices:
            # read sample inputs
            read_file_name = f"/ct_case_{idx + 1:04d}.csv"  # e.g. ct_case_0005
            record_inputs = np.loadtxt(self.inputs_data_path + read_file_name, delimiter=",")

            record_outputs = np.loadtxt(self.outputs_data_path + read_file_name, delimiter=",")

            edge: np.ndarray = generate_distance_based_edges(
                record_inputs[:, 0:3][np.newaxis, :, :], [0], self.sections, self.nodes_per_section
            )

            context_data = {
                "index": (idx, self.context_description["index"]),
                "points": (record_inputs.shape[0], self.context_description["points"]),
            }

            feature_data: Dict[str, Tuple[np.ndarray, str]] = {
                "node_coord": (record_inputs[:, 0:3], self.feature_description["node_coord"]),
                "laplace_coord": (record_inputs[:, 3:11], self.feature_description["laplace_coord"]),
                "fiber_and_sheet": (record_inputs[:, 11:17], self.feature_description["fiber_and_sheet"]),
                "edges_indices": (edge[0], self.feature_description["edges_indices"]),
                "mat_param": (data_global_feature[:, 1:7][idx], self.feature_description["mat_param"]),
                "pressure": (data_global_feature[:, 7:9][idx], self.feature_description["pressure"]),
                "shape_coeffs": (data_shape_coeff[:, 1:60][idx], self.feature_description["shape_coeffs"]),
                "displacement": (record_outputs[:, 0:3], self.feature_description["displacement"]),
                "stress": (record_outputs[:, 3:4], self.feature_description["stress"]),
            }

            writer.write(context_data, feature_data)  # noqa

            logger.info(f"index {idx} done")
            sys.stdout.flush()

        writer.close()

        return f"File {file_i}/{indices} written and closed"

    def _data_stats(self) -> None:
        # we only allow train data stats write to path
        write_to_path = self.data_type == TRAIN_NAME

        self._data_node_stats(write_to_path)

        self._data_global_feature_stats(write_to_path)

        self._data_label_stats(write_to_path)

        self._total_data_size()

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

    def _total_data_size(self) -> None:
        np.save(self.data_size_path, self.sample_indices.shape[0])
