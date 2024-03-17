from torch.utils.data import Dataset
from typing import Dict, Sequence
from pkg.utils.logging import init_logger
import os
import numpy as np
import torch

logger = init_logger("LvDataset")

nodes: str = "nodes"
edges: str = "edges"
shape_coeffs: str = "shape_coeffs"
theta_vals: str = "theta_vals"
displacement: str = "displacement"


class LvDataset(Dataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, task_data: Dict, data_type: str, n_shape_coeff: int = 2):
        base_data_path = f"{task_data['task_data_path']}/{task_data['sub_data_name']}"

        if not os.path.isdir(base_data_path):
            raise NotADirectoryError(f"No directory at: {base_data_path}")
        else:
            logger.info(f"base_data_path is {base_data_path}")

        processed_data_path = f"{base_data_path}/processedData/{data_type}"
        topology_data_path = f"{base_data_path}/topologyData"
        stats_data_path = f"{base_data_path}/normalisationStatistics"

        logger.info(f"processed_data_path is {processed_data_path}")
        logger.info(f"topology_data_path is {topology_data_path}")
        logger.info(f"stats_data_path is {stats_data_path}")

        # load mesh topology (assumed fixed for each graph)
        sparse_topology = np.load(f"{topology_data_path}/sparse-topology.npy").astype(np.int32)
        self._senders = sparse_topology[:, 0]
        self._receivers = sparse_topology[:, 1]

        # node features
        self._nodes = np.load(f"{processed_data_path}/augmented-node-features.npy").astype(np.float32)

        # edge features
        self._edges = np.load(f"{processed_data_path}/edge-features.npy").astype(np.float32)

        # array holding the displacement between end and start diastole
        # (normalised for training data, un-normalised for validation and test data)
        self._displacement = np.load(f"{processed_data_path}/real-node-displacement.npy").astype(np.float32)

        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(f"{stats_data_path}/real-node-displacement-mean.npy").astype(np.float32)
        self._displacement_std = np.load(f"{stats_data_path}/real-node-displacement-std.npy").astype(np.float32)

        # the co-ordinates of the LV geometry in the reference configuration
        self._real_node_coords = np.load(f"{processed_data_path}/real-node-coords.npy").astype(np.float32)

        node_layer_labels = np.load(f"{topology_data_path}/node-layer-labels.npy")

        self._real_node_indices = node_layer_labels == 0

        self._data_size = self._displacement.shape[0]
        self._output_dim = self._displacement.shape[-1]
        self._n_total_nodes = int(self._nodes.shape[1])
        self._n_real_nodes = self._real_node_indices.sum()
        self._n_edges = int(self._edges.shape[1])

        # array of data point indices that can be iterated over during each epoch
        self._epoch_indices = np.arange(self._data_size)

        self._n_shape_coeff = n_shape_coeff
        if self._n_shape_coeff > 0:
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

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(f"{processed_data_path}/global-features.npy").astype(np.float32)

        # summary statistics for global variables, calculated on training data
        self._theta_mean = np.load(f"{stats_data_path}/global-features-mean.npy").astype(np.float32)
        self._theta_std = np.load(f"{stats_data_path}/global-features-std.npy").astype(np.float32)

    def __len__(self):
        return self._data_size

    def __getitem__(self, index):
        sample = {
            nodes: self._nodes[index],
            edges: self._edges[index],
            shape_coeffs: self._shape_coeffs[index],
            theta_vals: self._theta_vals[index],
        }

        label = self._displacement[index]

        return sample, label

    def get_senders(self) -> torch.tensor:
        return torch.from_numpy(self._senders)

    def get_receivers(self) -> torch.tensor:
        return torch.from_numpy(self._receivers)

    def get_n_total_nodes(self) -> int:
        return self._n_total_nodes

    def get_real_node_indices(self) -> Sequence[bool]:
        return self._real_node_indices

    def get_displacement_mean(self) -> np.float32:
        return self._displacement_mean

    def get_displacement_std(self) -> np.float32:
        return self._displacement_std

