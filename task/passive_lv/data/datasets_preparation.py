import platform
from typing import Dict, Optional

import numpy as np
import pandas as pd
from numba.typed import List as Numba_List

from common.constant import TRAIN_NAME
from pkg.data_utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.train.datasets.base_datasets_preparation import AbstractDataPreparationDataset
from pkg.utils.io import check_and_clean_path
from task.passive_lv.data import logger
from task.passive_lv.data.datasets import FEHeartSageDataset


class FEHeartSagePreparationDataset(AbstractDataPreparationDataset, FEHeartSageDataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super(FEHeartSagePreparationDataset, self).__init__(data_config, data_type)

        self.edge_indices_generate_method = data_config["edge_indices_generate_method"]
        self.down_sampling: Optional[float] = (
            data_config.get("down_sampling", None) if self.data_type == TRAIN_NAME else None
        )
        self.select_nodes: Optional[np.ndarray] = None

    def prepare_dataset_process(self):
        if check_and_clean_path(self.dataset_path, self.overwrite_data):
            self._data_generation()
        else:
            logger.info(f"data already exists, no overwrite: {self.dataset_path}")

        self._data_down_sampling_node_selection()
        self._data_generation()

        if self.data_type == TRAIN_NAME and check_and_clean_path(self.stats_data_path, self.overwrite_stats):
            self._data_stats()

    def _data_generation(self):
        # fmt: off
        self._prepare_features()
        self._prepare_global_features("global_feature", self.theta_original_path, self.theta_path, np.float32, False)  # noqa
        self._prepare_global_features("shape_coeff", self.shape_coeff_original_path, self.shape_coeff_path, np.float32, False)  # noqa
        self._prepare_edge()
        # fmt: on

    def _data_stats(self):
        self._data_stats_total_size()

        # self._check_stats()

        self._prepare_node_coord_stats()

    def _data_down_sampling_node_selection(self):
        if self.down_sampling is None or self.down_sampling == 1.0:
            return

        data = np.load(self.displacement_original_path)
        node_size = data.shape[1]

        self.select_nodes = np.random.choice(node_size, size=int(node_size * self.down_sampling), replace=False)

        logger.info(f"given the down sampling ratio {self.down_sampling}, we choice {len(self.select_nodes)} nodes")

    def _prepare_features(self) -> None:
        node_features = np.load(self.node_feature_original_path).astype(np.float32)
        node_coord = np.load(self.node_coord_original_path).astype(np.float32)
        displacement = np.load(self.displacement_original_path).astype(np.float32)

        if self.down_sampling:
            logger.info(f"given the down sampling ratio {self.down_sampling}")

            num_samples, num_nodes, fea_dim = node_features.shape
            _, _, coord_dim = node_coord.shape
            _, _, displacement_dim = displacement.shape

            down_sample_node = int(num_nodes * self.down_sampling)

            new_node_features = np.empty((num_samples, down_sample_node, fea_dim), dtype=np.float32)
            new_node_coord = np.empty((num_samples, down_sample_node, coord_dim), dtype=np.float32)
            new_displacement = np.empty((num_samples, down_sample_node, displacement_dim), dtype=np.float32)

            for i in range(num_samples):
                select_nodes = np.random.choice(num_nodes, size=down_sample_node, replace=False)

                new_node_features[i] = node_features[i, select_nodes, :]
                new_node_coord[i] = node_coord[i, select_nodes, :]
                new_displacement[i] = displacement[i, select_nodes, :]

            node_features = new_node_features
            node_coord = new_node_coord
            displacement = new_displacement

        # === save node features
        np.save(self.node_feature_path, node_features)
        np.save(self.node_coord_path, node_coord)
        np.save(self.displacement_path, displacement)

        logger.info("====== prepare node and displacement DONE ======")

    def _prepare_global_features(
        self, fea_name: str, read_path: str, save_path: str, np_type: np.dtype, can_down_sampling: bool
    ) -> None:
        features = np.load(read_path).astype(np_type)

        if self.down_sampling and can_down_sampling:
            features = features[:, self.select_nodes, :]

        # === save node features
        np.save(save_path, features)

        logger.info(f"====== prepare {fea_name} DONE ======")

    def _prepare_edge(self):
        node_coords = np.load(f"{self.node_coord_path}").astype(np.float32)

        # edge features
        if self.edge_indices_generate_method == 0:  # old method from passive_lv_gnn_emul based on FEA + Kmeans node
            edges = self._calculate_edge_from_topology(self.topology_data_path)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif self.edge_indices_generate_method == 1:  # FEA node + 2 fixed node
            edges = self._calculate_edge_from_topology(self.topology_data_path)

            column1 = np.zeros((edges.shape[0], 1), np.int64)
            column2 = np.full((edges.shape[0], 1), 1500, np.int64)
            edges = np.concatenate((edges, column1, column2), axis=-1)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif self.edge_indices_generate_method == 2:  # FEA Node + fixed node selected based on points interval
            edges = self._calculate_edge_from_topology(self.topology_data_path)

            for i in range(1, 6700, 700):
                edge_column = np.full((edges.shape[0], 1), i, np.int64)
                edges = np.concatenate((edges, edge_column), axis=-1)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif self.edge_indices_generate_method == 3:  # fea Node + random node
            edges = self._calculate_edge_from_topology(self.topology_data_path)

            for i in range(5):
                edge_column = np.random.randint(low=1, high=6700, size=(edges.shape[0], 1))
                edges = np.concatenate((edges, edge_column), axis=-1)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif self.edge_indices_generate_method == 4:  # node based on the relative distance
            if platform.system() == "Darwin":  # macOS
                edges = self._calculate_node_neighbour_distance_ny(node_coords)
            else:
                edges = self._calculate_node_neighbour_distance_nb(node_coords)

        else:
            raise ValueError("please check and define the edge_generate_method properly")

        np.save(self.edge_file_path, edges.astype(np.int64))

        logger.info("====== prepare_edge DONE ======")

    def _calculate_edge_from_topology(self, data_path: str):
        from itertools import zip_longest

        node_layer_labels = np.load(f"{data_path}/node-layer-labels.npy")
        real_node_indices = np.where(node_layer_labels == 0)
        # load mesh topology (assumed fixed for each graph)
        sparse_topology = np.load(f"{data_path}/sparse-topology.npy").astype(np.int32)

        checked_topology_indices = np.all(np.isin(sparse_topology, real_node_indices), axis=1)
        real_topology_indices = sparse_topology[checked_topology_indices]
        reversed_topology_indices = real_topology_indices[:, ::-1]

        df_edge = pd.DataFrame(
            np.concatenate((real_topology_indices, reversed_topology_indices), axis=0), columns=["sender", "receiver"]
        )
        edge = df_edge.groupby("sender")["receiver"].apply(lambda x: sorted(list(set(x))))

        # Use groupby and apply a lambda function that converts data into a set.
        return np.array(list(map(list, zip_longest(*edge, fillvalue=self.default_padding_value)))).T

    def _calculate_node_neighbour_distance_nb(self, node_coord: np.ndarray) -> np.ndarray:
        num_samples = node_coord.shape[0]
        num_nodes = node_coord.shape[1]

        sections = self.sections
        nodes_per_section = self.nodes_per_sections

        sorted_indices_by_dist = np.empty((num_samples, num_nodes, sum(nodes_per_section)), dtype=np.int16)

        sections_nb = Numba_List()
        [sections_nb.append(x) for x in sections]

        nodes_per_section_nb = Numba_List()
        [nodes_per_section_nb.append(x) for x in nodes_per_section]

        for i in range(num_samples):
            sorted_indices_by_dist[i] = generate_distance_based_edges_nb(
                node_coord[i], sections_nb, nodes_per_section_nb
            )

            logger.info(f"calculate sorted_indices_by_dist for {i} done")

        return sorted_indices_by_dist

    # some issue with IOS OPM, will continue to use numpy to generate the neighbours
    def _calculate_node_neighbour_distance_ny(self, node_coord: np.ndarray) -> np.ndarray:
        num_samples = node_coord.shape[0]
        num_nodes = node_coord.shape[1]

        sections = self.sections
        nodes_per_section = self.nodes_per_sections

        sorted_indices_by_dist = np.empty((num_samples, num_nodes, sum(nodes_per_section)), dtype=np.int16)

        for i in range(num_samples):
            sorted_indices_by_dist[i] = generate_distance_based_edges_ny(node_coord, [i], sections, nodes_per_section)

            logger.info(f"calculate sorted_indices_by_dist for {i} done")

        return sorted_indices_by_dist

    def _prepare_node_coord_stats(self):
        node_coords = np.load(self.node_coord_path).astype(np.float32)

        coord_max_norm_val = np.max(node_coords, axis=(0, 1))
        coord_min_norm_val = np.min(node_coords, axis=(0, 1))

        np.save(self.node_coord_max_path, coord_max_norm_val)
        np.save(self.node_coord_min_path, coord_min_norm_val)

        logger.info(
            f"{self.data_type} prepare_node_stats preset max_norm and min_norm is "
            f"{coord_max_norm_val} {coord_min_norm_val} "
        )

    def _check_stats(self):
        from pkg.data_utils.stats import stats_analysis

        node_coords = np.load(self.node_coord_original_path).astype(np.float32)

        stats_analysis("node_coord", node_coords, 2, "", logger, False)  # noqa

    def _data_stats_total_size(self):
        data_size = np.load(self.displacement_path).astype(np.float32).shape[0]

        np.save(self.data_size_path, data_size)

        logger.info(f"{self.data_type} dataset prepare_data_size_stats preset size is {data_size}")
