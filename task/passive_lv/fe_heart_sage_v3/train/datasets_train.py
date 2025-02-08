from typing import Dict

import numpy as np
import torch

from common.constant import TRAIN_NAME
from pkg.train.datasets.base_datasets_train import BaseDataset
from task.passive_lv.data import logger
from task.passive_lv.data.datasets import FEPassiveLVHeartDataset

normal_norm = "normal_norm"
max_min_norm = "max_min_norm"


class FEHeartSageTrainDataset(BaseDataset, FEPassiveLVHeartDataset):
    """Dataset class for training a graph neural network on finite element passive left ventricle heart data.

    This class handles loading and preprocessing of node features, coordinates, edge indices,
    shape coefficients, and displacement data for training. It inherits from BaseDataset and
    FEPassiveLVHeartDataset to provide dataset functionality.
    """

    def __init__(self, data_config: Dict, data_type: str) -> None:
        """Initialize the training dataset.

        Args:
            data_config (Dict): Configuration dictionary containing dataset parameters
            data_type (str): Type of dataset (train/val/test)
        """
        super().__init__(data_config, data_type)

        self.device = "cuda" if self.gpu else "cpu"

        self.n_shape_coeff = data_config.get("n_shape_coeff", 2)

        # node features (used real node features)
        # === fetch node features
        node_features = np.load(self.node_feature_path).astype(np.float32)
        node_coords = np.load(self.node_coord_path).astype(np.float32)

        # === max min calculation
        self._coord_max = np.load(self.node_coord_max_path)
        self._coord_min = np.load(self.node_coord_min_path)

        self._node_features = node_features
        self._node_coord = self.normalization_max_min(node_coords, self._coord_max, self._coord_min)
        logger.info(f"node_features shape: {self._node_features.shape}, node_coord: {self._node_coord.shape}")
        logger.info(f"_coord_max: {self._coord_max}, _coord_min: {self._coord_min}")

        # edge features
        self._edges_indices = np.load(self.edge_file_path).astype(np.int64)
        logger.info(f"edges shape: {self._edges_indices.shape}")

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        # summary statistics for displacement values calculated on training data
        self._theta_max = np.load(self.theta_max_path).astype(np.float32)
        self._theta_min = np.load(self.theta_min_path).astype(np.float32)

        theta_vals = np.load(self.theta_path).astype(np.float32)
        self._theta_vals = self.normalization_max_min(theta_vals, self._theta_max, self._theta_min)
        logger.info(f"_theta_max: {self._theta_max}, _theta_min: {self._theta_min}")

        # labels
        # summary statistics for displacement values calculated on training data
        self._displacement_max = np.load(self.displacement_max_path).astype(np.float32)
        self._displacement_min = np.load(self.displacement_min_path).astype(np.float32)

        displacement = np.load(self.displacement_path).astype(np.float32)

        if self.data_type == TRAIN_NAME:
            logger.info(f"data type = {self.data_type}, need to normalize displacement")
            self._displacement = self.normalization_max_min(
                displacement, self._displacement_max, self._displacement_min
            )
        else:
            logger.info(f"data type = {self.data_type}, no need to normalize displacement")
            self._displacement = displacement
        logger.info(f"_displacement_max: {self._displacement_max}, _displacement_min: {self._displacement_min}")

        if self.n_shape_coeff > 0:
            # load shape coefficients
            shape_coeffs = np.load(self.shape_coeff_path).astype(np.float32)

            assert shape_coeffs.shape[-1] >= self.n_shape_coeff, (
                f"Number of shape coefficients to retain "
                f"({self.n_shape_coeff}) must be <= number of columns in "
                f"'shape-coeffs.npy' ({shape_coeffs.shape[-1]})"
            )

            # retain n_shape_coeff of these to input to the emulator
            self._shape_coeff_max = np.load(self.shape_coeff_max_path).astype(np.float32)
            self._shape_coeff_min = np.load(self.shape_coeff_min_path).astype(np.float32)

            self._shape_coeffs = self.normalization_max_min(shape_coeffs, self._shape_coeff_max, self._shape_coeff_min)[
                :, : self.n_shape_coeff
            ]
            logger.info(f"_shape_coeff_max: {self._shape_coeff_max}, _shape_coeff_min: {self._shape_coeff_min}")

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

    def __getitem__(self, index) -> (Dict, torch.Tensor):
        node_features = self._node_features[index]
        node_coord = self._node_coord[index]
        edges_indices = self._edges_indices[index]
        shape_coeffs = self._shape_coeffs[index]
        theta_vals = self._theta_vals[index]

        displacement = self._displacement[index]

        selected_node_num = 300

        selected_node = (
            torch.randint(0, node_coord.shape[1], size=(selected_node_num,), dtype=torch.int64, device=self.device)
            .unsqueeze(0)
            .expand(node_coord.shape[0], -1)
        )

        sample = {
            "node_features": node_features,
            "node_coord": node_coord,
            "edges_indices": edges_indices,
            "shape_coeffs": shape_coeffs,
            "theta_vals": theta_vals,
            "selected_node": selected_node,
        }

        labels = {"displacement": displacement}

        return sample, labels

    def get_displacement_max(self) -> torch.tensor:
        """Get the max displacement value for normalization.

        Returns:
            torch.tensor: max displacement value, moved to GPU if using CUDA
        """
        _displacement_max = torch.from_numpy(self._displacement_max)
        return _displacement_max if not self.gpu else _displacement_max.cuda()

    def get_displacement_min(self) -> torch.tensor:
        """Get the min of displacement values for normalization.

        Returns:
            torch.tensor: min displacement values, moved to GPU if using CUDA
        """
        _displacement_min = torch.from_numpy(self._displacement_min)
        return _displacement_min if not self.gpu else _displacement_min.cuda()

    @staticmethod
    def normalization_max_min(array: np.ndarray, max_val: float, min_val: float) -> np.ndarray:
        """Normalize coordinate values using min-max normalization."""
        return (array - min_val) / (max_val - min_val)

    def displacement_normalization_max_min(self, array: np.ndarray) -> np.ndarray:
        """Normalize coordinate values using min-max normalization.

        Args:
            array (np.ndarray): Array of coordinate values to normalize

        Returns:
            np.ndarray: Normalized coordinate values between 0 and 1
        """
        return (array - self._displacement_min) / (self._displacement_max - self._displacement_min)
