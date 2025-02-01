import os
from typing import Dict, Sequence

import numpy as np
import torch

from pkg.train.datasets.base_datasets_train import BaseDataset
from pkg.utils.logs import init_logger

logger = init_logger("LVDataset")


class LvDataset(BaseDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology.

    This class handles loading and processing of Left Ventricle (LV) mesh data for graph neural networks.
    It manages node features, edge features, displacement data, and various other mesh-related attributes.

    Attributes:
        _senders (np.ndarray): Source nodes in the graph topology
        _receivers (np.ndarray): Target nodes in the graph topology
        _nodes (torch.Tensor): Node features for the mesh
        _edges (torch.Tensor): Edge features for the mesh
        _displacement (torch.Tensor): Displacement between end and start diastole
        _displacement_mean (np.ndarray): Mean of displacement values from training data
        _displacement_std (np.ndarray): Standard deviation of displacement values
        _real_node_coords (np.ndarray): Coordinates of LV geometry in reference configuration
        _real_node_indices (np.ndarray): Boolean mask for real nodes
        _data_size (int): Total number of samples in the dataset
        _n_total_nodes (int): Total number of nodes in the mesh
        _shape_coeffs (torch.Tensor): Shape coefficients for the mesh
        _theta_vals (torch.Tensor): Global material stiffness parameters
        _theta_mean (np.ndarray): Mean of global parameters from training data
        _theta_std (np.ndarray): Standard deviation of global parameters
    """

    def __init__(self, data_config: Dict, data_type: str) -> None:
        """Initialize the LV dataset with configuration and data type.

        Args:
            data_config (Dict): Configuration dictionary containing paths and parameters
            data_type (str): Type of dataset (train/val/test)

        Raises:
            NotADirectoryError: If the specified base data path doesn't exist
        """
        super().__init__(data_config, data_type)

        base_data_path = f"{data_config['task_data_path']}"
        n_shape_coeff = data_config["n_shape_coeff"]

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
        self._nodes = torch.from_numpy(np.load(f"{processed_data_path}/augmented-node-features.npy").astype(np.float32))

        # edge features
        self._edges = torch.from_numpy(np.load(f"{processed_data_path}/edge-features.npy").astype(np.float32))

        # array holding the displacement between end and start diastole
        # (normalised for training data, un-normalised for validation and test data)
        self._displacement = torch.from_numpy(
            np.load(f"{processed_data_path}/real-node-displacement.npy").astype(np.float32)
        )

        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(f"{stats_data_path}/real-node-displacement-mean.npy").astype(np.float32)
        self._displacement_std = np.load(f"{stats_data_path}/real-node-displacement-std.npy").astype(np.float32)

        # the co-ordinates of the LV geometry in the reference configuration
        self._real_node_coords = np.load(f"{processed_data_path}/real-node-coords.npy").astype(np.float32)

        node_layer_labels = np.load(f"{topology_data_path}/node-layer-labels.npy")

        self._real_node_indices = node_layer_labels == 0

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
        """Get the sender nodes of the graph topology.

        Returns:
            torch.tensor: Array of sender node indices, moved to GPU if self.gpu is True
        """
        _senders = torch.from_numpy(self._senders)
        return _senders if not self.gpu else _senders.cuda()

    def get_receivers(self) -> torch.tensor:
        """Get the receiver nodes of the graph topology.

        Returns:
            torch.tensor: Array of receiver node indices, moved to GPU if self.gpu is True
        """
        _receivers = torch.from_numpy(self._receivers)
        return _receivers if not self.gpu else _receivers.cuda()

    def get_n_total_nodes(self) -> int:
        """Get the total number of nodes in the mesh.

        Returns:
            int: Total number of nodes
        """
        return self._n_total_nodes

    def get_real_node_indices(self) -> Sequence[torch.tensor]:
        """Get boolean mask for real nodes in the mesh.

        Returns:
            Sequence[torch.tensor]: Boolean mask indicating real nodes, moved to GPU if self.gpu is True
        """
        _real_node_indices = torch.from_numpy(self._real_node_indices)
        return _real_node_indices if not self.gpu else _real_node_indices.cuda()

    def get_displacement_mean(self) -> torch.tensor:
        """Get the mean displacement values from training data.

        Returns:
            torch.tensor: Mean displacement values, moved to GPU if self.gpu is True
        """
        _displacement_mean = torch.from_numpy(self._displacement_mean)
        return self._displacement_mean if not self.gpu else _displacement_mean.cuda()

    def get_displacement_std(self) -> torch.tensor:
        """Get the standard deviation of displacement values from training data.

        Returns:
            torch.tensor: Standard deviation of displacement values, moved to GPU if self.gpu is True
        """
        _displacement_std = torch.from_numpy(self._displacement_std)
        return _displacement_std if not self.gpu else _displacement_std.cuda()

    def get_head_inputs(self, batch_size) -> Dict:
        """Get input dictionary for the first batch_size samples.

        Args:
            batch_size (int): Number of samples to include in the batch

        Returns:
            Dict: Dictionary containing input features for the specified batch size
        """
        inputs, _ = self.__getitem__(np.arange(0, batch_size))

        return inputs
