import os
from typing import Dict, Sequence, Optional

import numpy as np
import torch

from pkg.train.datasets.base_datasets import BaseDataset
from pkg.utils.logging import init_logger

logger = init_logger("LV_Dataset")


class LvDataset(BaseDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self,
                 data_config: Dict,
                 data_type: str,
                 corrd_max_norm_val: Optional[np.array] = None,
                 corrd_min_norm_val: Optional[np.array] = None,
                 distance_max_norm_val: Optional[np.array] = None,
                 distance_min_norm_val: Optional[np.array] = None,
                 n_shape_coeff: int = 2
        ) -> None:
        super().__init__(data_config, data_type)

        self.coord_max_norm_val, self.coord_min_norm_val, self.distance_max_norm_val, self.distance_min_norm_val = (
            corrd_max_norm_val, corrd_min_norm_val, distance_max_norm_val, distance_min_norm_val
        )

        base_data_path = f"{data_config['task_data_path']}"

        if not os.path.isdir(base_data_path):
            raise NotADirectoryError(f"No directory at: {base_data_path}")
        else:
            logger.info(f"base_data_path is {base_data_path}")

        raw_data_path = f"{base_data_path}/rawData/{data_type}"
        processed_data_path = f"{base_data_path}/processedData/{data_type}"
        topology_data_path = f"{base_data_path}/topologyData"
        stats_data_path = f"{base_data_path}/normalisationStatistics"

        logger.info(f"processed_data_path is {processed_data_path}")
        logger.info(f"topology_data_path is {topology_data_path}")
        logger.info(f"stats_data_path is {stats_data_path}")

        # node features (used real node features)
        # === fetch node features
        node_is_edge = torch.from_numpy(np.load(f"{raw_data_path}/real-node-features.npy").astype(np.float32))
        node_coords = np.load(f"{processed_data_path}/real-node-coords.npy").astype(np.float32)

        # === distance calculation
        node_distance = np.expand_dims(np.sqrt((node_coords ** 2).sum(axis=2)), axis=2)

        # === max min calculation
        if (self.coord_max_norm_val is None and self.coord_min_norm_val is None and
                self.distance_max_norm_val is None and self.distance_min_norm_val is None):
            self.coord_max_norm_val = np.max(node_coords, axis=(0, 1))
            self.coord_min_norm_val = np.min(node_coords, axis=(0, 1))
            self.distance_max_norm_val = np.max(node_distance, axis=(0, 1))
            self.distance_min_norm_val = np.min(node_distance, axis=(0, 1))
        else:
            logger.info(f"{data_type} dataset preset max_norm and min_norm is "
                        f"{self.coord_max_norm_val} {self.coord_min_norm_val} "
                        f"{self.distance_max_norm_val} {self.distance_min_norm_val}"
            )

        self._nodes = torch.from_numpy(np.concatenate((
            node_is_edge,
            self.coord_normalization_max_min(node_coords),
            self.distance_normalization_max_min(node_distance)
             ), axis=2)
        )

        # edge features
        # === real node indices
        node_layer_labels = np.load(f"{topology_data_path}/node-layer-labels.npy")
        real_node_indices = np.where(node_layer_labels == 0)

        # load mesh topology (assumed fixed for each graph)
        sparse_topology = np.load(f"{topology_data_path}/sparse-topology.npy").astype(np.int32)

        checked_topology_indices = np.all(np.isin(sparse_topology, real_node_indices), axis=1)
        real_topology_indices = sparse_topology[checked_topology_indices]
        self._senders = real_topology_indices[:, 0]
        self._receivers = real_topology_indices[:, 1]

        # ==== calculate edge features
        edge = self.generate_topology_data(real_topology_indices, node_coords)

        # === calculate edge distance
        edge_distance = np.expand_dims(np.sqrt((edge ** 2).sum(axis=2)), axis=2)

        self._edges = torch.from_numpy(np.concatenate((edge, edge_distance), axis=2))

        # array holding the displacement between end and start diastole
        # (normalised for training data, un-normalised for validation and test data)
        self._displacement = torch.from_numpy(np.load(f"{processed_data_path}/real-node-displacement.npy").astype(np.float32))

        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(f"{stats_data_path}/real-node-displacement-mean.npy").astype(np.float32)
        self._displacement_std = np.load(f"{stats_data_path}/real-node-displacement-std.npy").astype(np.float32)

        self._data_size = self._displacement.shape[0]
        self._n_total_nodes = int(self._nodes.shape[1])

        if n_shape_coeff > 0:
            # load shape coefficients
            shape_coeffs = np.load(f"{processed_data_path}/shape-coeffs.npy").astype(np.float32)

            assert shape_coeffs.shape[-1] >= n_shape_coeff, (
                f"Number of shape coefficients to retain "
                f"({n_shape_coeff}) must be <= number of columns in "
                f"'shape-coeffs.npy' ({shape_coeffs.shape[-1]})"
            )

            # retain n_shape_coeff of these to input to the emulator
            self._shape_coeffs = shape_coeffs[:, :n_shape_coeff]

        else:
            self._shape_coeffs = [None] * self._data_size

        self._shape_coeffs = torch.from_numpy(self._shape_coeffs)

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = torch.from_numpy(np.load(f"{processed_data_path}/global-features.npy").astype(np.float32))

        # summary statistics for global variables, calculated on training data
        self._theta_mean = np.load(f"{stats_data_path}/global-features-mean.npy").astype(np.float32)
        self._theta_std = np.load(f"{stats_data_path}/global-features-std.npy").astype(np.float32)

        if self.gpu:
            self._nodes = self._nodes.cuda()
            self._edges = self._edges.cuda()
            self._shape_coeffs = self._shape_coeffs.cuda()
            self._theta_vals = self._theta_vals.cuda()
            self._displacement = self._displacement.cuda()

    def __len__(self):
        return self._data_size

    def __getitem__(self, index) -> (Dict, torch.Tensor):
        nodes = self._nodes[index]
        edges = self._edges[index]
        shape_coeffs = self._shape_coeffs[index]
        theta_vals = self._theta_vals[index]

        labels = self._displacement[index]

        sample = {
            "nodes": nodes,
            "edges": edges,
            "shape_coeffs": shape_coeffs,
            "theta_vals": theta_vals,
        }

        return sample, labels

    def get_senders(self) -> torch.tensor:
        _senders = torch.from_numpy(self._senders)
        return _senders if not self.gpu else _senders.cuda()

    def get_receivers(self) -> torch.tensor:
        _receivers = torch.from_numpy(self._receivers)
        return _receivers if not self.gpu else _receivers.cuda()

    def get_n_total_nodes(self) -> int:
        return self._n_total_nodes

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
