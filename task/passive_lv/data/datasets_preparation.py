import platform
from typing import Dict, Optional

import numpy as np
import pandas as pd
from numba.typed import List as Numba_List

from pkg.data_utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.train.datasets.base_datasets_preparation import AbstractDataPreparationDataset
from task.passive_lv.data import logger
from task.passive_lv.data.datasets import FEPassiveLVHeartDataset


class FEPassiveLVHeartPreparationDataset(AbstractDataPreparationDataset, FEPassiveLVHeartDataset):
    """Dataset class for preparing Finite Element Passive Left Ventricle Heart data.

    This class handles data preparation tasks like loading raw data, preprocessing features,
    generating edge indices, and computing statistics for the dataset.
    """

    def __init__(self, data_config: Dict, data_type: str) -> None:
        """Initialize the dataset with configuration parameters.

        Args:
            data_config: Dictionary containing dataset configuration parameters
            data_type: String indicating the type of dataset (train/val/test)
        """
        super(FEPassiveLVHeartPreparationDataset, self).__init__(data_config, data_type)

        logger.info(f"=== Init FEPassiveLVHeartPreparationDataset {data_type} data config start ===")

        self.edge_indices_generate_method = data_config["edge_indices_generate_method"]
        self.sections = data_config["sections"]
        self.nodes_per_sections = data_config["nodes_per_sections"]
        self.default_padding_value = data_config.get("default_padding_value", -1)

        # self.down_sampling: Optional[float] = (
        #     data_config.get("down_sampling", None) if self.data_type == TRAIN_NAME else None
        # )
        self.select_nodes: Optional[np.ndarray] = None

        # logger.info(f"edge_indices_generate_method is {self.edge_indices_generate_method}")

        logger.info(f"=== Init FEPassiveLVHeartPreparationDataset {data_type} data config done ===")

    def _data_generation(self):
        # fmt: off

        self._prepare_features()

        self._prepare_global_features("global_feature", self.theta_original_path, self.theta_path, np.float32)  # noqa

        self._prepare_global_features("shape_coeff", self.shape_coeff_original_path, self.shape_coeff_path, np.float32)  # noqa

        self._prepare_edge()

        # fmt: on

    def _data_stats(self):
        self._data_stats_total_size()

        # self._check_stats()

        self._prepare_node_coord_stats()

        self._prepare_node_displacement_stats()

    def _prepare_features(self) -> None:
        """Prepare and save node features, coordinates and displacement data.

        Loads raw data, processes it and saves in the required format.
        """
        logger.info("====== prepare node and displacement start ======")

        node_features = np.load(self.node_feature_original_path).astype(np.float32)
        node_coord = np.load(self.node_coord_original_path).astype(np.float32)
        displacement = np.load(self.displacement_original_path).astype(np.float32)
        raw_displacement = np.load(self.displacement_raw_original_path).astype(np.float32)

        np.save(self.node_feature_path, node_features)
        np.save(self.node_coord_path, node_coord)
        np.save(self.raw_displacement_path, raw_displacement)
        np.save(self.displacement_path, displacement)

        logger.info("====== prepare node and displacement done ======")

    @staticmethod
    def _prepare_global_features(fea_name: str, read_path: str, save_path: str, np_type: np.dtype) -> None:
        """Prepare and save global features.

        Args:
            fea_name: Name of the feature
            read_path: Path to read raw data from
            save_path: Path to save processed data to
            np_type: Numpy dtype for the data
        """
        logger.info(f"====== prepare {fea_name} start ======")

        features = np.load(read_path).astype(np_type)
        np.save(save_path, features)

        logger.info(f"====== prepare {fea_name} done ======")

    def _prepare_edge(self):
        """Prepare and save edge indices based on specified method.

        Supports multiple edge generation methods including topology-based and distance-based approaches.
        """
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
        """Calculate edge indices from topology data.

        Args:
            data_path: Path to topology data files

        Returns:
            Array of edge indices based on mesh topology
        """
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

        return np.array(list(map(list, zip_longest(*edge, fillvalue=self.default_padding_value)))).T

    def _calculate_node_neighbour_distance_nb(self, node_coord: np.ndarray) -> np.ndarray:
        """Calculate edge indices based on node distances using Numba implementation.

        Args:
            node_coord: Node coordinates array

        Returns:
            Array of edge indices based on node distances
        """
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

    def _calculate_node_neighbour_distance_ny(self, node_coord: np.ndarray) -> np.ndarray:
        """Calculate edge indices based on node distances using NumPy implementation.

        Args:
            node_coord: Node coordinates array

        Returns:
            Array of edge indices based on node distances
        """
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
        """Calculate and save node coordinate statistics.

        Computes and saves max and min values for normalization.
        """
        node_coords = np.load(self.node_coord_path).astype(np.float32)

        coord_max_norm_val = np.max(node_coords, axis=(0, 1))
        coord_min_norm_val = np.min(node_coords, axis=(0, 1))

        np.save(self.node_coord_max_path, coord_max_norm_val)
        np.save(self.node_coord_min_path, coord_min_norm_val)

        logger.info(
            f"{self.data_type} prepare_node_stats preset max_norm and min_norm is "
            f"{coord_max_norm_val} {coord_min_norm_val} "
        )

    def _prepare_node_displacement_stats(self):
        """Calculate and save node coordinate statistics.

        Computes and saves max and min values for normalization.
        """
        node_displacement = np.load(self.displacement_original_path).astype(np.float32)

        displacement_max_norm_val = np.max(node_displacement, axis=(0, 1))
        displacement_min_norm_val = np.min(node_displacement, axis=(0, 1))

        np.save(self.displacement_max_path, displacement_max_norm_val)
        np.save(self.displacement_min_path, displacement_min_norm_val)

        logger.info(
            f"{self.data_type} prepare_displacement_stats preset max_norm and min_norm is "
            f"{displacement_max_norm_val} {displacement_min_norm_val} "
        )

    def _check_stats(self):
        """Perform statistical analysis on node coordinates."""
        from pkg.data_utils.stats import stats_analysis

        node_coords = np.load(self.node_coord_original_path).astype(np.float32)

        stats_analysis("node_coord", node_coords, 2, "", logger, False)  # noqa

    def _data_stats_total_size(self):
        """Calculate and save total dataset size."""
        data_size = np.load(self.displacement_path).astype(np.float32).shape[0]

        np.save(self.data_size_path, data_size)

        logger.info(f"{self.data_type} dataset prepare_data_size_stats preset size is {data_size}")
