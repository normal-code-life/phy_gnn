import os
from typing import Dict

import numpy as np
import pandas as pd
from numba.typed import List as Numba_List

from pkg.data.utils.edge_generation import (generate_distance_based_edges_nb,
                                            generate_distance_based_edges_ny)
from pkg.utils.logs import init_logger
from task.passive_lv.fe_heart_sage_v1.data.datasets import FEHeartSageV1Dataset, logger
import platform


class FEHeartSageV1PreparationDataset(FEHeartSageV1Dataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)

        if os.path.exists(self.edge_file_path):
            logger.info("the data is already exist. no longer need data preparation")

        node_coords = np.load(f"{self.processed_data_path}/real-node-coords.npy").astype(np.float32)

        # edge features
        edge_indices_generate_method = data_config["edge_indices_generate_method"]
        if edge_indices_generate_method == 0:  # old method from passive_lv_gnn_emul based on FEA node + Kmeans node
            edges = self._calculate_edge_from_topology(self.topology_data_path)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif edge_indices_generate_method == 1:  # FEA node + 2 fixed node
            edges = self._calculate_edge_from_topology(self.topology_data_path)

            column1 = np.zeros((edges.shape[0], 1), np.int64)
            column2 = np.full((edges.shape[0], 1), 1500, np.int64)
            edges = np.concatenate((edges, column1, column2), axis=-1)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif edge_indices_generate_method == 2:  # FEA Node + fixed node selected based on points interval
            edges = self._calculate_edge_from_topology(self.topology_data_path)

            for i in range(1, 6700, 700):
                edge_column = np.full((edges.shape[0], 1), i, np.int64)
                edges = np.concatenate((edges, edge_column), axis=-1)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif edge_indices_generate_method == 3:  # fea Node + random node
            edges = self._calculate_edge_from_topology(self.topology_data_path)

            for i in range(5):
                edge_column = np.random.randint(low=1, high=6700, size=(edges.shape[0], 1))
                edges = np.concatenate((edges, edge_column), axis=-1)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif edge_indices_generate_method == 4:  # node based on the relative distance
            if platform.system() == "Darwin":  # macOS
                edges = self._calculate_node_neighbour_distance_ny(node_coords)
            else:
                edges = self._calculate_node_neighbour_distance_nb(node_coords)

        else:
            raise ValueError("please check and define the edge_generate_method properly")

        np.save(self.edge_file_path, edges.astype(np.int64))

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
            sorted_indices_by_dist[i] = generate_distance_based_edges_ny(
                node_coord, [i], sections, nodes_per_section
            )

            logger.info(f"calculate sorted_indices_by_dist for {i} done")

        return sorted_indices_by_dist
