from typing import Dict

import numpy as np
import torch

from pkg.train.datasets.base_datasets_train import BaseDataset
from task.passive_lv.data import logger
from task.passive_lv.data.datasets import FEPassiveLVHeartDataset


class FEHeartSageTrainDataset(BaseDataset, FEPassiveLVHeartDataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)

        self.device = "cuda" if self.gpu else "cpu"

        self.n_shape_coeff = data_config.get("n_shape_coeff", 2)

        # node features (used real node features)
        # === fetch node features
        node_features = np.load(self.node_feature_path).astype(np.float32)
        node_coords = np.load(self.node_coord_path).astype(np.float32)

        # === distance calculation
        # node_distance = np.expand_dims(np.sqrt((node_coords ** 2).sum(axis=2)), axis=2)

        # === max min calculation
        self.coord_max_norm_val = np.load(self.node_coord_max_path)
        self.coord_min_norm_val = np.load(self.node_coord_min_path)

        self._node_features = node_features
        self._node_coord = self.coord_normalization_max_min(node_coords)
        logger.info(f"node_features shape: {self._node_features.shape}, node_coord: {self._node_coord.shape}")

        # edge features
        self._edges_indices = np.load(self.edge_file_path).astype(np.int64)
        logger.info(f"edges shape: {self._edges_indices.shape}")

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(self.theta_path).astype(np.float32)

        # labels
        self._displacement = np.load(self.displacement_path).astype(np.float32)

        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(self.displacement_mean_path).astype(np.float32)
        self._displacement_std = np.load(self.displacement_std_path).astype(np.float32)

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

        # if self.gpu:
        #     node_features = node_features.cuda()
        #     node_coord = node_coord.cuda()
        #     edges_indices = edges_indices.cuda()
        #     theta_vals = theta_vals.cuda()
        #     shape_coeffs = shape_coeffs.cuda()
        #     displacement = displacement.cuda()

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

    def get_displacement_mean(self) -> torch.tensor:
        _displacement_mean = torch.from_numpy(self._displacement_mean)
        return self._displacement_mean if not self.gpu else _displacement_mean.cuda()

    def get_displacement_std(self) -> torch.tensor:
        _displacement_std = torch.from_numpy(self._displacement_std)
        return _displacement_std if not self.gpu else _displacement_std.cuda()

    def coord_normalization_max_min(self, array: np.ndarray) -> np.ndarray:
        # max_val = np.expand_dims(self.coord_max_norm_val, axis=(0, 1))
        # min_val = np.expand_dims(self.coord_min_norm_val, axis=(0, 1))

        max_val = max(self.coord_max_norm_val)
        min_val = min(self.coord_min_norm_val)

        return (array - min_val) / (max_val - min_val)
