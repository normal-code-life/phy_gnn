import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from pkg.train.datasets.base_datasets import BaseDataset
from pkg.utils.logging import init_logger

logger = init_logger("GraphSage_Dataset")


class GraphSageDataset(BaseDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(
        self,
        data_config: Dict,
        data_type: str,
        coord_max_norm_val: Optional[np.array] = None,
        coord_min_norm_val: Optional[np.array] = None,
        distance_max_norm_val: Optional[np.array] = None,
        distance_min_norm_val: Optional[np.array] = None,
    ) -> None:
        super().__init__(data_config, data_type)

        self.coord_max_norm_val, self.coord_min_norm_val, self.distance_max_norm_val, self.distance_min_norm_val = (
            coord_max_norm_val,
            coord_min_norm_val,
            distance_max_norm_val,
            distance_min_norm_val,
        )

        self.default_padding_value = data_config.get("default_padding_value", -1)
        self.n_shape_coeff = data_config.get("n_shape_coeff", 2)

        # fetch data from local path
        base_data_path = f"{data_config['task_data_path']}"
        base_task_path = f"{data_config['task_path']}"

        if not os.path.isdir(base_data_path):
            raise NotADirectoryError(f"No directory at: {base_data_path}")
        else:
            logger.info(f"base_data_path is {base_data_path}")

        raw_data_path = f"{base_data_path}/rawData/{data_type}"
        processed_data_path = f"{base_data_path}/processedData/{data_type}"
        topology_data_path = f"{base_data_path}/topologyData"
        stats_data_path = f"{base_data_path}/normalisationStatistics"

        logger.info(f"base_data_path is {base_data_path}")
        logger.info(f"base_task_path is {base_task_path}")
        logger.info(f"processed_data_path is {processed_data_path}")
        logger.info(f"topology_data_path is {topology_data_path}")
        logger.info(f"stats_data_path is {stats_data_path}")

        # node features (used real node features)
        # === fetch node features
        node_features = np.load(f"{raw_data_path}/real-node-features.npy").astype(np.float32)
        node_coords = np.load(f"{processed_data_path}/real-node-coords.npy").astype(np.float32)

        # === distance calculation
        # node_distance = np.expand_dims(np.sqrt((node_coords ** 2).sum(axis=2)), axis=2)

        # === max min calculation
        if (
            self.coord_max_norm_val is None
            and self.coord_min_norm_val is None
            and self.distance_max_norm_val is None
            and self.distance_min_norm_val is None
        ):
            self.coord_max_norm_val = np.max(node_coords, axis=(0, 1))
            self.coord_min_norm_val = np.min(node_coords, axis=(0, 1))
            # self.distance_max_norm_val = np.max(node_distance, axis=(0, 1))
            # self.distance_min_norm_val = np.min(node_distance, axis=(0, 1))
        else:
            logger.info(
                f"{data_type} dataset preset max_norm and min_norm is "
                f"{self.coord_max_norm_val} {self.coord_min_norm_val} "
                f"{self.distance_max_norm_val} {self.distance_min_norm_val}"
            )

        self._node_features = node_features
        self._node_coord = self.coord_normalization_max_min(node_coords)
        logger.info(f"node_features shape: {self._node_features.shape}, node_coord: {self._node_coord.shape}")

        # edge features
        edge_indices_generate_method = data_config["edge_indices_generate_method"]
        if edge_indices_generate_method == 0:
            edges = self._calculate_edge_from_topology(topology_data_path)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)
        elif edge_indices_generate_method == 1:
            edges = self._calculate_edge_from_topology(topology_data_path)

            # column1 = np.zeros((edges.shape[0], 1), np.int64)
            # column2 = np.full((edges.shape[0], 1), 1500, np.int64)
            # edges = np.concatenate((edges, column1, column2), axis=-1)
            # edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

            # for i in range(1, 6700, 700):
            #     edge_column = np.full((edges.shape[0], 1), i, np.int64)
            #     edges = np.concatenate((edges, edge_column), axis=-1)
            # edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

            for i in range(5):
                edge_column = np.random.randint(low=1, high=6700, size=(edges.shape[0], 1))
                edges = np.concatenate((edges, edge_column), axis=-1)
            edges = np.repeat(edges[np.newaxis, :, :], node_coords.shape[0], axis=0)

        elif edge_indices_generate_method == 2:
            edge_file_path = f"{processed_data_path}/node_neighbours_distance_{data_type}_9_3.npy"
            if os.path.exists(edge_file_path):
                edges = np.load(edge_file_path).astype(np.float32)
            else:
                edges = self._calculate_node_neighbour_distance(node_coords)
                np.save(edge_file_path, edges)
            edges = edges.astype(np.int64)
        else:
            raise ValueError("please check and define the edge_generate_method properly")

        self._edges_indices = edges
        logger.info(f"edges shape: {self._edges_indices.shape}")

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(f"{processed_data_path}/global-features.npy").astype(np.float32)

        # summary statistics for global variables, calculated on training data
        self._theta_mean = np.load(f"{stats_data_path}/global-features-mean.npy").astype(np.float32)
        self._theta_std = np.load(f"{stats_data_path}/global-features-std.npy").astype(np.float32)

        # labels
        self._displacement = np.load(f"{processed_data_path}/real-node-displacement.npy").astype(np.float32)

        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(f"{stats_data_path}/real-node-displacement-mean.npy").astype(np.float32)
        self._displacement_std = np.load(f"{stats_data_path}/real-node-displacement-std.npy").astype(np.float32)

        self._data_size = self._displacement.shape[0]

        if self.n_shape_coeff > 0:
            # load shape coefficients
            shape_coeffs = np.load(f"{processed_data_path}/shape-coeffs.npy").astype(np.float32)

            assert shape_coeffs.shape[-1] >= self.n_shape_coeff, (
                f"Number of shape coefficients to retain "
                f"({self.n_shape_coeff}) must be <= number of columns in "
                f"'shape-coeffs.npy' ({shape_coeffs.shape[-1]})"
            )

            # retain n_shape_coeff of these to input to the emulator
            self._shape_coeffs = shape_coeffs[:, : self.n_shape_coeff]

        else:
            self._shape_coeffs = [None] * self._data_size

        self._node_features = torch.from_numpy(self._node_features)
        self._node_coord = torch.from_numpy(self._node_coord)
        self._edges_indices = torch.from_numpy(self._edges_indices)
        self._theta_vals = torch.from_numpy(self._theta_vals)
        self._displacement = torch.from_numpy(self._displacement)
        self._shape_coeffs = torch.from_numpy(self._shape_coeffs)

        if self.gpu:
            self._node_features = self._node_features.cuda()
            self._node_coord = self._node_coord.cuda()
            self._edges_indices = self._edges_indices.cuda()
            self._theta_vals = self._theta_vals.cuda()
            self._shape_coeffs = self._shape_coeffs.cuda()
            self._displacement = self._displacement.cuda()

    def __len__(self):
        return self._data_size

    def __getitem__(self, index) -> (Dict, torch.Tensor):
        node_features = self._node_features[index]
        node_coord = self._node_coord[index]
        edges_indices = self._edges_indices[index]
        shape_coeffs = self._shape_coeffs[index]
        theta_vals = self._theta_vals[index]

        labels = self._displacement[index]

        sample = {
            "node_features": node_features,
            "node_coord": node_coord,
            "edges_indices": edges_indices,
            "shape_coeffs": shape_coeffs,
            "theta_vals": theta_vals,
        }

        return sample, labels

    # method 0: read original default topology data
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

    def _calculate_node_neighbour_distance(self, node_coord: np.ndarray, batch_size: int = 20) -> np.ndarray:
        num_nodes = node_coord.shape[0]
        sorted_indices_by_dist = np.empty((num_nodes, node_coord.shape[1], 100), dtype=np.int16)
        for i in range(0, num_nodes, batch_size):
            end = min(i + batch_size, num_nodes)
            relative_positions = node_coord[i:end, :, np.newaxis, :] - node_coord[i:end, np.newaxis, :, :]
            relative_distance = np.sqrt(np.sum(np.square(relative_positions), axis=-1, keepdims=True))
            sorted_indices = np.argsort(relative_distance.squeeze(axis=-1), axis=-1)
            sorted_indices_by_dist[i:end] = self._random_select_nodes(sorted_indices[..., 1:1001])

            logger.info(f"calculate sorted_indices_by_dist for {i} done")

        return sorted_indices_by_dist  # remove the node itself

    def _random_select_nodes(self, indices: np.ndarray) -> np.ndarray:
        batch_size, rows, cols = indices.shape
        sections = [0, 60, 100, 200, 500, cols]
        max_select_node = [50, 20, 15, 10, 5]
        num_select_total = sum(max_select_node)

        selected_indices = np.zeros((batch_size, rows, num_select_total), dtype=np.int32)

        for i in range(len(sections) - 1):
            start_idx = 0 if i == 0 else sum(max_select_node[:i])
            num_random_indices = max_select_node[i]
            range_start = sections[i]
            range_end = sections[i + 1]

            for b in range(batch_size):
                for r in range(rows):
                    random_indices = np.random.permutation(range_end - range_start)[:num_random_indices]
                    selected_indices[b, r, start_idx:start_idx + num_random_indices] = random_indices + range_start

        # Gather the selected indices from the original indices
        batch_indices = np.arange(batch_size)[:, None, None]
        row_indices = np.arange(rows)[None, :, None]
        selected_values = indices[batch_indices, row_indices, selected_indices]

        return selected_values

    def _calculate_edge_by_top_k(self, sorted_indices_by_dist: np.ndarray, k: int) -> np.ndarray:
        return sorted_indices_by_dist[..., :k]

    def get_displacement_mean(self) -> torch.tensor:
        _displacement_mean = torch.from_numpy(self._displacement_mean)
        return self._displacement_mean if not self.gpu else _displacement_mean.cuda()

    def get_displacement_std(self) -> torch.tensor:
        _displacement_std = torch.from_numpy(self._displacement_std)
        return _displacement_std if not self.gpu else _displacement_std.cuda()

    def generate_topology_data(self, topology_indices: np.ndarray, node_coord: np.ndarray) -> np.ndarray:
        shape = np.shape(node_coord)

        senders = node_coord[np.arange(shape[0])[:, None], topology_indices[:, 0], :]
        receivers = node_coord[np.arange(shape[0])[:, None], topology_indices[:, 1], :]

        return senders - receivers

    def coord_normalization_max_min(self, array: np.ndarray) -> np.ndarray:
        max_val = np.expand_dims(self.coord_max_norm_val, axis=(0, 1))
        min_val = np.expand_dims(self.coord_min_norm_val, axis=(0, 1))

        return (array - min_val) / (max_val - min_val)

    def distance_normalization_max_min(self, array: np.ndarray) -> np.ndarray:
        max_val = np.expand_dims(self.distance_max_norm_val, axis=(0, 1))
        min_val = np.expand_dims(self.distance_min_norm_val, axis=(0, 1))

        return (array - min_val) / (max_val - min_val)

    def get_distance_max_norm_val(self) -> np.array:
        return self.distance_max_norm_val

    def get_distance_min_norm_val(self) -> np.array:
        return self.distance_min_norm_val

    def get_coord_max_norm_val(self) -> np.array:
        return self.coord_max_norm_val

    def get_coord_min_norm_val(self) -> np.array:
        return self.coord_min_norm_val


# # edge features
# # === real node indices
# node_layer_labels = np.load(f"{topology_data_path}/node-layer-labels.npy")
# real_node_indices = np.where(node_layer_labels == 0)
#
# # load mesh topology (assumed fixed for each graph)
# sparse_topology = np.load(f"{topology_data_path}/sparse-topology.npy").astype(np.int32)
#
# checked_topology_indices = np.all(np.isin(sparse_topology, real_node_indices), axis=1)
# real_topology_indices = sparse_topology[checked_topology_indices]
# self._senders = real_topology_indices[:, 0]
# self._receivers = real_topology_indices[:, 1]
#
# # ==== calculate edge features
# edge = self.generate_topology_data(real_topology_indices, node_coords)
#
# # === calculate edge distance
# edge_distance = np.expand_dims(np.sqrt((edge ** 2).sum(axis=2)), axis=2)
#
# self._edges = torch.from_numpy(np.concatenate((edge, edge_distance), axis=2))

# array holding the displacement between end and start diastole
# (normalised for training data, un-normalised for validation and test data)
