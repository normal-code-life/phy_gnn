from typing import Dict

import numpy as np
import torch

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

        self.displacement_norm_type = data_config.get("displacement_norm_type", normal_norm)
        if self.displacement_norm_type != normal_norm and self.displacement_norm_type != max_min_norm:
            raise ValueError("please assign the right value for displacement_norm_type")

        self.n_shape_coeff = data_config.get("n_shape_coeff", 2)

        # node features (used real node features)
        # === fetch node features
        node_features = np.load(self.node_feature_path).astype(np.float32)
        node_coords = np.load(self.node_coord_path).astype(np.float32)

        # === max min calculation
        self._coord_max = np.load(self.node_coord_max_path)
        self._coord_min = np.load(self.node_coord_min_path)

        self._node_features = node_features
        self._node_coord = self.coord_normalization_max_min(node_coords)
        logger.info(f"node_features shape: {self._node_features.shape}, node_coord: {self._node_coord.shape}")

        # edge features
        self._edges_indices = np.load(self.edge_file_path).astype(np.int64)
        logger.info(f"edges shape: {self._edges_indices.shape}")

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(self.theta_path).astype(np.float32)

        # labels
        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(self.displacement_mean_path).astype(np.float32)
        self._displacement_std = np.load(self.displacement_std_path).astype(np.float32)
        self._displacement_max = np.load(self.displacement_max_path).astype(np.float32)
        self._displacement_min = np.load(self.displacement_min_path).astype(np.float32)

        if self.displacement_norm_type == normal_norm:
            self._displacement = np.load(self.displacement_path).astype(np.float32)
        elif self.displacement_norm_type == max_min_norm:
            displacement = np.load(self.raw_displacement_path).astype(np.float32)
            self._displacement = self.displacement_normalization_max_min(displacement)

        if self.n_shape_coeff > 0:
            # load shape coefficients
            shape_coeffs = np.load(self.shape_coeff_path).astype(np.float32)

            assert shape_coeffs.shape[-1] >= self.n_shape_coeff, (
                f"Number of shape coefficients to retain "
                f"({self.n_shape_coeff}) must be <= number of columns in "
                f"'shape-coeffs.npy' ({shape_coeffs.shape[-1]})"
            )

            # retain n_shape_coeff of these to input to the emulator
            self._shape_coeffs = shape_coeffs[:, : self.n_shape_coeff]

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

        # for version `fe_heart_sage_v3` we need to input extra `selected_node` and `selected_node_num`:
        # using this strategy is only a transitional solution, it will be overwritten later on.
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
        """Get the mean displacement value for normalization.

        Returns:
            torch.tensor: Mean displacement value, moved to GPU if using CUDA
        """
        _displacement_max = torch.from_numpy(self._displacement_max)
        return _displacement_max if not self.gpu else _displacement_max.cuda()

    def get_displacement_min(self) -> torch.tensor:
        """Get the standard deviation of displacement values for normalization.

        Returns:
            torch.tensor: Standard deviation of displacement values, moved to GPU if using CUDA
        """
        _displacement_min = torch.from_numpy(self._displacement_min)
        return _displacement_min if not self.gpu else _displacement_min.cuda()

    def get_displacement_mean(self) -> torch.tensor:
        """Get the mean displacement value for normalization.

        Returns:
            torch.tensor: Mean displacement value, moved to GPU if using CUDA
        """
        _displacement_mean = torch.from_numpy(self._displacement_mean)
        return self._displacement_mean if not self.gpu else _displacement_mean.cuda()

    def get_displacement_std(self) -> torch.tensor:
        """Get the standard deviation of displacement values for normalization.

        Returns:
            torch.tensor: Standard deviation of displacement values, moved to GPU if using CUDA
        """
        _displacement_std = torch.from_numpy(self._displacement_std)
        return _displacement_std if not self.gpu else _displacement_std.cuda()

    def coord_normalization_max_min(self, array: np.ndarray) -> np.ndarray:
        """Normalize coordinate values using min-max normalization.

        Args:
            array (np.ndarray): Array of coordinate values to normalize

        Returns:
            np.ndarray: Normalized coordinate values between 0 and 1
        """
        max_val = max(self._coord_max)
        min_val = min(self._coord_min)

        return (array - min_val) / (max_val - min_val)

    def displacement_normalization_max_min(self, array: np.ndarray) -> np.ndarray:
        """Normalize coordinate values using min-max normalization.

        Args:
            array (np.ndarray): Array of coordinate values to normalize

        Returns:
            np.ndarray: Normalized coordinate values between 0 and 1
        """
        max_val = max(self._displacement_max)
        min_val = min(self._displacement_min)

        return (array - min_val) / (max_val - min_val)
