from typing import Dict, Optional

import numpy as np
import torch

from pkg.utils.logs import init_logger
from task.passive_lv.fe_heart_sage_v1.data.datasets import FEHeartSageV1Dataset

logger = init_logger("FEHeartSage_Dataset")


class FEHeartSageV1TrainDataset(FEHeartSageV1Dataset):
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

        # node features (used real node features)
        # === fetch node features
        node_features = np.load(self.node_feature_path).astype(np.float32)
        node_coords = np.load(self.node_coord_path).astype(np.float32)

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
                f"{self.data_type} dataset preset max_norm and min_norm is "
                f"{self.coord_max_norm_val} {self.coord_min_norm_val} "
                f"{self.distance_max_norm_val} {self.distance_min_norm_val}"
            )

        self._node_features = node_features
        self._node_coord = self.coord_normalization_max_min(node_coords)
        logger.info(f"node_features shape: {self._node_features.shape}, node_coord: {self._node_coord.shape}")

        # edge features
        self._edges_indices = np.load(self.edge_file_path).astype(np.int64)
        logger.info(f"edges shape: {self._edges_indices.shape}")

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(self.theta_path).astype(np.float32)

        # summary statistics for global variables, calculated on training data
        self._theta_mean = np.load(f"{self.stats_data_path}/global-features-mean.npy").astype(np.float32)
        self._theta_std = np.load(f"{self.stats_data_path}/global-features-std.npy").astype(np.float32)

        # labels
        self._displacement = np.load(self.displacement_path).astype(np.float32)

        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(f"{self.stats_data_path}/real-node-displacement-mean.npy").astype(np.float32)
        self._displacement_std = np.load(f"{self.stats_data_path}/real-node-displacement-std.npy").astype(np.float32)

        self._data_size = self._displacement.shape[0]

        if self.n_shape_coeff > 0:
            # load shape coefficients
            shape_coeffs = np.load(self.n_shape_coeff_path).astype(np.float32)

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

    def get_displacement_mean(self) -> torch.tensor:
        _displacement_mean = torch.from_numpy(self._displacement_mean)
        return self._displacement_mean if not self.gpu else _displacement_mean.cuda()

    def get_displacement_std(self) -> torch.tensor:
        _displacement_std = torch.from_numpy(self._displacement_std)
        return _displacement_std if not self.gpu else _displacement_std.cuda()

    def coord_normalization_max_min(self, array: np.ndarray) -> np.ndarray:
        max_val = np.expand_dims(self.coord_max_norm_val, axis=(0, 1))
        min_val = np.expand_dims(self.coord_min_norm_val, axis=(0, 1))

        return (array - min_val) / (max_val - min_val)
