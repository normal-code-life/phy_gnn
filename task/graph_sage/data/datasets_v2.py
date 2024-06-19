import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from pkg.train.datasets.base_datasets import BaseIterableDataset
from pkg.utils.logging import init_logger

logger = init_logger("GraphSage_Dataset")


class GraphSageDataset(BaseIterableDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)

        # fetch data from local path
        base_data_path = f"{data_config['task_data_path']}"
        base_task_path = f"{data_config['task_path']}"

        self.exp_name = data_config.get("exp_name", None)
        self.default_padding_value = data_config.get("default_padding_value", -1)
        self.n_shape_coeff = data_config.get("n_shape_coeff", 2)
        self.chunk_size = data_config.get("chunk_size", 50)

        if not os.path.isdir(base_data_path):
            raise NotADirectoryError(f"No directory at: {base_data_path}")
        else:
            logger.info(f"base_data_path is {base_data_path}")

        self.raw_data_path = f"{base_data_path}/rawData/{data_type}"
        self.processed_data_path = f"{base_data_path}/processedData/{data_type}"
        self.topology_data_path = f"{base_data_path}/topologyData"
        self.stats_data_path = f"{base_data_path}/normalisationStatistics"
        self.pt_data_path = f"{base_data_path}/ptData"

        logger.info(f"base_data_path is {base_data_path}")
        logger.info(f"base_task_path is {base_task_path}")
        logger.info(f"processed_data_path is {self.processed_data_path}")
        logger.info(f"topology_data_path is {self.topology_data_path}")
        logger.info(f"stats_data_path is {self.stats_data_path}")

        # node
        self.real_node_features_path = f"{self.raw_data_path}/real-node-features.npy"
        self.real_node_coord_path = f"{self.raw_data_path}/real-node-coords.npy"
        self.coord_max_norm_path = f"{self.stats_data_path}/real_node_coord_max_norm.npy"
        self.coord_min_norm_path = f"{self.stats_data_path}/real_node_coord_min_norm.npy"

        # edge
        self.edge_file_path = f"{self.processed_data_path}/node_neighbours_all_distance_{self.data_type}.npy"

        # theta
        self.theta_vals_path = f"{self.processed_data_path}/global-features.npy"
        self.theta_mean_path = f"{self.stats_data_path}/global-features-mean.npy"
        self.theta_std_path = f"{self.stats_data_path}/global-features-std.npy"

        # displacement
        self.displacement_path = f"{self.processed_data_path}/real-node-displacement.npy"
        self.displacement_mean_path = f"{self.stats_data_path}/real-node-displacement-mean.npy"
        self.displacement_std_path = f"{self.stats_data_path}/real-node-displacement-std.npy"

        self.coeffs_path = f"{self.processed_data_path}/shape-coeffs.npy"


class GraphSagePreprocessDataset(GraphSageDataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super(GraphSagePreprocessDataset, self).__init__(data_config, data_type)

    def preprocess(self):
        self.preprocess_node_neighbours()

        self.preprocess_node_coord()

    def preprocess_node_neighbours(self) -> None:
        file_path = self.edge_file_path

        if os.path.exists(file_path):
            return

        if self.gpu:
            self._preprocess_edge_by_gpu(file_path)
        else:
            self._preprocess_edge_by_cpu(file_path)

        return

    def _preprocess_edge_by_cpu(self, file_path: str) -> None:
        node = np.load(self.real_node_coord_path, mmap_mode="r").astype(np.float32)

        node_shape = node.shape

        with open(file_path, "wb") as f:
            for i in range(0, node_shape[0], self.chunk_size):
                end = min(i + self.chunk_size, node_shape[0])

                relative_positions = node[i:end, :, np.newaxis, :] - node[i:end, np.newaxis, :, :]

                relative_distance = np.sqrt(np.sum(np.square(relative_positions), axis=-1, keepdims=True))

                sorted_indices = np.argsort(relative_distance.squeeze(axis=-1), axis=-1)

                chunk = sorted_indices[..., 1:].astype(np.int32)

                chunk.tofile(f)

                logger.info(f"calculate sorted_indices_by_dist for {i} - {i + self.chunk_size - 1} done")

    def _preprocess_edge_by_gpu(self, file_path: str) -> None:
        node = np.load(self.real_node_coord_path, mmap_mode="r").astype(np.float32)

        node_shape = node.shape

        node = torch.tensor(node, device="cuda")  # 将节点数据移到 GPU

        with open(file_path, "wb") as f:
            for i in range(0, node_shape[0], self.chunk_size):
                end = min(i + self.chunk_size, node_shape[0])

                relative_positions = node[i:end, :, None, :] - node[i:end, None, :, :]  # 计算相对位置

                relative_distance = torch.sqrt(torch.sum(torch.square(relative_positions), dim=-1, keepdim=True))

                sorted_indices = torch.argsort(relative_distance.squeeze(-1), dim=-1)  # 排序

                chunk = sorted_indices[..., 1:].to(torch.int32).cpu().numpy()  # 将结果转换为 int 并移回 CPU

                chunk.tofile(f)  # 写入文件

                logger.info(f"calculate sorted_indices_by_dist for {i} - {i + self.chunk_size - 1} done")

    def preprocess_node_coord(self) -> None:
        node_coords = np.load(f"{self.processed_data_path}/real-node-coords.npy", mmap_mode="r").astype(np.float32)

        coord_max_norm_val = np.max(node_coords, axis=(0, 1))
        coord_min_norm_val = np.min(node_coords, axis=(0, 1))

        np.save(self.coord_max_norm_path, coord_max_norm_val)
        np.save(self.coord_min_norm_path, coord_min_norm_val)


class GraphSageTrainDataset(GraphSageDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)

        # node features/coord (used real node features)
        # === coord max min calculation
        node_coords = np.load(self.real_node_coord_path, mmap_mode="r").astype(np.float32)

        self.coord_max_norm_val = np.load(self.coord_max_norm_path).astype(np.float32)
        self.coord_min_norm_val = np.load(self.coord_min_norm_path).astype(np.float32)
        logger.info(
            f"{self.data_type} dataset max and min norm is " f"{self.coord_max_norm_val} {self.coord_min_norm_val}"
        )

        self._node_coords = self._normal_max_min_transform(node_coords, self.coord_max_norm_val, self.coord_min_norm_val)
        self._node_features = np.load(self.real_node_features_path, mmap_mode="r").astype(np.float32)

        logger.info(f"node_features shape: {self._node_features.shape}, node_coord: {self._node_coords.shape}")

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(self.theta_vals_path, mmap_mode="r").astype(np.float32)
        logger.info(f"theta vals shape: {self._theta_vals.shape}")

        # labels
        self._displacement = np.load(self.displacement_path, mmap_mode="r").astype(np.float32)
        logger.info(f"displacement shape: {self._displacement.shape}")

        # summary statistics for displacement values calculated on training data
        self.displacement_mean = np.load(self.displacement_mean_path).astype(np.float32)
        self.displacement_std = np.load(self.displacement_std_path).astype(np.float32)

        self._shape_coeffs = np.load(self.coeffs_path, mmap_mode="r").astype(np.float32)[:, : self.n_shape_coeff]
        logger.info(f"shape_coeffs shape: {self._shape_coeffs.shape}")

        self.data_size, self.node_size, _ = self._displacement.shape

        # edge features
        self._edges_indices = np.memmap(self.edge_file_path, dtype=np.int32, mode='r', shape=(self.data_size, self.node_size, self.node_size - 1))
        logger.info(f"edges shape: {self._edges_indices.shape}")

        assert (
                self._node_coords.shape[0]
                == self._node_features.shape[0]
                == self._edges_indices.shape[0]
                == self._theta_vals.shape[0]
                == self._displacement.shape[0]
                == self._shape_coeffs.shape[0]
        ), (
            f"Variables are not equal: "
            f"node_coords.shape[0]={self._node_coords.shape[0]}, "
            f"node_features.shape[0]={self._node_features.shape[0]}, "
            f"edges_indices.shape[0]={self._edges_indices.shape[0]}, "
            f"theta_vals.shape[0]={self._theta_vals.shape[0]}, "
            f"displacement.shape[0]={self._displacement.shape[0]} "
            f"shape_coeffs.shape[0]={self._shape_coeffs.shape[0]}"
        )

        assert (node_coords.shape[1] ==
                self._node_features.shape[1] ==
                self._edges_indices.shape[1] ==
                self._displacement.shape[1]
                ), (
            f"Variables are not equal: "
            f"node_coords.shape[0]={self._node_coords.shape[0]}, "
            f"node_features.shape[0]={self._node_features.shape[0]}, "
            f"edges_indices.shape[0]={self._edges_indices.shape[0]}, "
            f"displacement.shape[0]={self._displacement.shape[0]} "
        )

    def __len__(self):
        return self.data_size

    def __iter__(self) -> (Dict, torch.Tensor):

        for sample in zip(
                self._node_coords, self._node_features, self._edges_indices,
                self._theta_vals, self._shape_coeffs, self._displacement
        ):

            node_coord = torch.from_numpy(sample[0])
            node_features = torch.from_numpy(sample[1])
            edges_indices = torch.from_numpy(sample[2])
            theta_vals = torch.from_numpy(sample[3])
            shape_coeffs = torch.from_numpy(sample[4])

            labels = torch.from_numpy(sample[5])

            if self.gpu:
                node_coord = node_coord.cuda()
                node_features = node_features.cuda()
                edges_indices = edges_indices.cuda()
                theta_vals = theta_vals.cuda()
                shape_coeffs = shape_coeffs.cuda()

                labels = labels.cuda()

            yield {
                "node_coord": node_coord,
                "node_features": node_features,
                "edges_indices": edges_indices,
                "shape_coeffs": shape_coeffs,
                "theta_vals": theta_vals,
            }, labels

    def _normal_max_min_transform(
            self, array: np.ndarray, max_norm_val: np.float32, min_norm_val: np.float32
    ) -> np.ndarray:
        max_val = np.expand_dims(max_norm_val, axis=(0, 1))
        min_val = np.expand_dims(min_norm_val, axis=(0, 1))

        return (array - min_val) / (max_val - min_val)

    def get_displacement_mean(self) -> torch.tensor:
        _displacement_mean = torch.from_numpy(self.displacement_mean)
        return self.displacement_mean if not self.gpu else _displacement_mean.cuda()

    def get_displacement_std(self) -> torch.tensor:
        _displacement_std = torch.from_numpy(self.displacement_std)
        return _displacement_std if not self.gpu else _displacement_std.cuda()
