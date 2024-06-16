import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from pkg.train.datasets.base_datasets import BaseDataset
from pkg.utils.logging import init_logger

logger = init_logger("GraphSage_Dataset")

norm_stats_file = "normalisationStatistics"
real_node_coord_max_norm = "real_node_coord_max_norm"
real_node_coord_min_norm = "real_node_coord_min_norm"


class GraphSageDataset(BaseDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)
        # fetch data from local path
        base_data_path = f"{data_config['task_data_path']}"
        base_task_path = f"{data_config['task_path']}"

        self.exp_name = data_config.get("exp_name", None)
        self.default_padding_value = data_config.get("default_padding_value", -1)
        self.n_shape_coeff = data_config.get("n_shape_coeff", 2)

        if not os.path.isdir(base_data_path):
            raise NotADirectoryError(f"No directory at: {base_data_path}")
        else:
            logger.info(f"base_data_path is {base_data_path}")

        self.raw_data_path = f"{base_data_path}/rawData/{data_type}"
        self.processed_data_path = f"{base_data_path}/processedData/{data_type}"
        self.topology_data_path = f"{base_data_path}/topologyData"
        self.stats_data_path = f"{base_data_path}/{norm_stats_file}"

        logger.info(f"base_data_path is {base_data_path}")
        logger.info(f"base_task_path is {base_task_path}")
        logger.info(f"processed_data_path is {self.processed_data_path}")
        logger.info(f"topology_data_path is {self.topology_data_path}")
        logger.info(f"stats_data_path is {self.stats_data_path}")

        # node features (used real node features)
        # === fetch node features
        node_features = np.load(f"{self.raw_data_path}/real-node-features.npy").astype(np.float32)
        node_coords = np.load(f"{self.processed_data_path}/real-node-coords.npy").astype(np.float32)

        # === distance calculation
        # node_distance = np.expand_dims(np.sqrt((node_coords ** 2).sum(axis=2)), axis=2)

        # === max min calculation
        coord_max_norm_path = f"{self.stats_data_path}/{real_node_coord_max_norm}.npy"
        coord_min_norm_path = f"{self.stats_data_path}/{real_node_coord_min_norm}.npy"

        if os.path.exists(coord_max_norm_path) and os.path.exists(coord_max_norm_path):
            self._coord_max_norm_val = np.load(coord_max_norm_path).astype(np.float32)
            self._coord_min_norm_val = np.load(coord_min_norm_path).astype(np.float32)

            logger.info(
                f"{self.data_type} dataset preset max_norm and min_norm is "
                f"{self._coord_max_norm_val} {self._coord_min_norm_val}"
            )

            self._node_features = node_features
            self._node_coord = self._coord_normalization_max_min(node_coords)
            logger.info(f"node_features shape: {self._node_features.shape}, node_coord: {self._node_coord.shape}")

        # edge features
        edge_file_path = f"{self.processed_data_path}/node_neighbours_all_distance_{self.data_type}.npy"
        if os.path.exists(edge_file_path):
            self._edges_indices = np.load(edge_file_path, mmap_mode="r").astype(np.int64)
            logger.info(f"edges shape: {self._edges_indices.shape}")

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(f"{self.processed_data_path}/global-features.npy").astype(np.float32)

        # summary statistics for global variables, calculated on training data
        self._theta_mean = np.load(f"{self.stats_data_path}/global-features-mean.npy").astype(np.float32)
        self._theta_std = np.load(f"{self.stats_data_path}/global-features-std.npy").astype(np.float32)

        # labels
        self._displacement = np.load(f"{self.processed_data_path}/real-node-displacement.npy").astype(np.float32)

        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(f"{self.stats_data_path}/real-node-displacement-mean.npy").astype(np.float32)
        self._displacement_std = np.load(f"{self.stats_data_path}/real-node-displacement-std.npy").astype(np.float32)

        self._data_size = self._displacement.shape[0]

        if self.n_shape_coeff > 0:
            # load shape coefficients
            shape_coeffs = np.load(f"{self.processed_data_path}/shape-coeffs.npy").astype(np.float32)

            assert shape_coeffs.shape[-1] >= self.n_shape_coeff, (
                f"Number of shape coefficients to retain "
                f"({self.n_shape_coeff}) must be <= number of columns in "
                f"'shape-coeffs.npy' ({shape_coeffs.shape[-1]})"
            )

            # retain n_shape_coeff of these to input to the emulator
            self._shape_coeffs = shape_coeffs[:, : self.n_shape_coeff]

        else:
            self._shape_coeffs = [None] * self._data_size

    def __len__(self):
        return self._data_size

    def __getitem__(self, index) -> (Dict, torch.Tensor):
        node_features = torch.from_numpy(self._node_features[index])
        node_coord = torch.from_numpy(self._node_coord[index])
        edges_indices = torch.from_numpy(self._edges_indices[index])
        theta_vals = torch.from_numpy(self._theta_vals[index])
        shape_coeffs = torch.from_numpy(self._shape_coeffs[index])

        labels = torch.from_numpy(self._displacement[index])

        if self.gpu:
            node_features = node_features.cuda()
            node_coord = node_coord.cuda()
            edges_indices = edges_indices.cuda()
            theta_vals = theta_vals.cuda()
            shape_coeffs = shape_coeffs.cuda()

            labels = labels.cuda()

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

    def _coord_normalization_max_min(self, array: np.ndarray) -> np.ndarray:
        max_val = np.expand_dims(self._coord_max_norm_val, axis=(0, 1))
        min_val = np.expand_dims(self._coord_min_norm_val, axis=(0, 1))

        return (array - min_val) / (max_val - min_val)

    def preprocess_node_neighbours(self, chunk_size: int = 50) -> None:
        file_path = f"{self.processed_data_path}/node_neighbours_all_distance_{self.data_type}.npy"

        if os.path.exists(file_path):
            return

        if self.gpu:
            self._preprocess_edge_by_gpu(file_path, chunk_size)
        else:
            self._preprocess_edge_by_cpu(file_path, chunk_size)

        return

    def _preprocess_edge_by_cpu(self, file_path: str, chunk_size: int) -> None:
        node = np.load(f"{self.processed_data_path}/real-node-coords.npy").astype(np.float32)

        node_shape = node.shape

        with open(file_path, 'wb') as f:
            for i in range(0, node_shape[0], chunk_size):
                end = min(i + chunk_size, node_shape[0])

                relative_positions = node[i:end, :, np.newaxis, :] - node[i:end, np.newaxis, :, :]

                relative_distance = np.sqrt(np.sum(np.square(relative_positions), axis=-1, keepdims=True))

                sorted_indices = np.argsort(relative_distance.squeeze(axis=-1), axis=-1)

                chunk = sorted_indices[..., 1:].astype(np.int32)

                chunk.tofile(f)

                logger.info(f"calculate sorted_indices_by_dist for {i} - {i + chunk_size - 1} done")

    def _preprocess_edge_by_gpu(self, file_path: str, chunk_size: int) -> None:
        node = np.load(f"{self.processed_data_path}/real-node-coords.npy").astype(np.float32)

        node_shape = node.shape

        node = torch.tensor(node, device='cuda')  # 将节点数据移到 GPU

        with open(file_path, 'wb') as f:
            for i in range(0, node_shape[0], chunk_size):
                end = min(i + chunk_size, node_shape[0])

                relative_positions = node[i: end, :, None, :] - node[i: end, None, :, :]  # 计算相对位置

                relative_distance = torch.sqrt(torch.sum(torch.square(relative_positions), dim=-1, keepdim=True))

                sorted_indices = torch.argsort(relative_distance.squeeze(-1), dim=-1)  # 排序

                chunk = sorted_indices[..., 1:].to(torch.int32).cpu().numpy()  # 将结果转换为 int 并移回 CPU

                chunk.tofile(f)  # 写入文件

                logger.info(f"calculate sorted_indices_by_dist for {i} - {i + chunk_size - 1} done")

    def preprocess_node_coord(self) -> None:
        node_coords = np.load(f"{self.processed_data_path}/real-node-coords.npy").astype(np.float32)

        coord_max_norm_val = np.max(node_coords, axis=(0, 1))
        coord_min_norm_val = np.min(node_coords, axis=(0, 1))

        np.save(f"{self.stats_data_path}/{real_node_coord_max_norm}", coord_max_norm_val)
        np.save(f"{self.stats_data_path}/{real_node_coord_min_norm}", coord_min_norm_val)

