import os
import random
from typing import Dict, Tuple
from torchvision import transforms
from pkg.train.module.data_transform import TFRecordToTensor, MaxMinNormalize, max_val_name, mim_val_name, TensorToGPU, CovertToModelInputs
from task.graph_sage_v2.data.data_transform import ConvertDataDim
import numpy as np
import torch
import torch.utils.data
from pkg.train.datasets.base_datasets import BaseIterableDataset
from pkg.utils.logging import init_logger
import tfrecord
from tfrecord.torch.dataset import MultiTFRecordDataset
import math

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
        self.chunk_file_size = data_config.get("chunk_file_size", 10)

        if not os.path.isdir(base_data_path):
            raise NotADirectoryError(f"No directory at: {base_data_path}")
        else:
            logger.info(f"base_data_path is {base_data_path}")

        self.raw_data_path = f"{base_data_path}/rawData/{self.data_type}"
        self.processed_data_path = f"{base_data_path}/processedData/{self.data_type}"
        self.topology_data_path = f"{base_data_path}/topologyData"
        self.stats_data_path = f"{base_data_path}/normalisationStatistics"
        self.tfrecord_path = f"{base_data_path}/tfData/{self.data_type}"

        logger.info(f"base_data_path is {base_data_path}")
        logger.info(f"base_task_path is {base_task_path}")
        logger.info(f"processed_data_path is {self.processed_data_path}")
        logger.info(f"tfrecord_path is {self.tfrecord_path}")
        logger.info(f"topology_data_path is {self.topology_data_path}")
        logger.info(f"stats_data_path is {self.stats_data_path}")

        # node
        self.real_node_features_path = f"{self.raw_data_path}/real-node-features.npy"
        self.real_node_coord_path = f"{self.processed_data_path}/real-node-coords.npy"
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

        # others
        self.data_size_path = f"{self.processed_data_path}/{self.data_type}_data_size.npy"

        # features
        self.context_description: Dict[str, str] = {
            "index": "int",
        }

        self.feature_description: Dict[str, str] = {
            "node_coord": "float",
            "node_features": "float",
            "edges_indices": "int",
            "shape_coeffs": "float",
            "theta_vals": "float",
            "labels": "float",
        }


class GraphSagePreprocessDataset(GraphSageDataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super(GraphSagePreprocessDataset, self).__init__(data_config, data_type)

    def preprocess(self):
        self._preprocess_data()
        self._data_stats()

    def _preprocess_data(self):
        file_path = self.tfrecord_path

        if len(os.listdir(file_path)) > 0:
            return

        # node features/coord (used real node features)
        self._node_features = np.load(self.real_node_features_path).astype(np.float32)

        self._node_coords = np.load(f"{self.processed_data_path}/real-node-coords.npy").astype(np.float32)

        logger.info(f"node_features shape: {self._node_features.shape}, node_coord: {self._node_coords.shape}")

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(self.theta_vals_path).astype(np.float32)
        logger.info(f"theta vals shape: {self._theta_vals.shape}")

        # labels
        self._displacement = np.load(self.displacement_path).astype(np.float32)
        logger.info(f"displacement shape: {self._displacement.shape}")

        self._shape_coeffs = np.load(self.coeffs_path, mmap_mode="r").astype(np.float32)[:, : self.n_shape_coeff]
        logger.info(f"shape_coeffs shape: {self._shape_coeffs.shape}")

        assert (
                self._node_coords.shape[0]
                == self._node_features.shape[0]
                == self._theta_vals.shape[0]
                == self._displacement.shape[0]
                == self._shape_coeffs.shape[0]
        ), (
            f"Variables are not equal: "
            f"node_coords.shape[0]={self._node_coords.shape[0]}, "
            f"node_features.shape[0]={self._node_features.shape[0]}, "
            f"theta_vals.shape[0]={self._theta_vals.shape[0]}, "
            f"displacement.shape[0]={self._displacement.shape[0]} "
            f"shape_coeffs.shape[0]={self._shape_coeffs.shape[0]}"
        )

        assert (self._node_coords.shape[1]
                == self._node_features.shape[1]
                == self._displacement.shape[1]
                ), (
            f"Variables are not equal: "
            f"node_coords.shape[0]={self._node_coords.shape[0]}, "
            f"node_features.shape[0]={self._node_features.shape[0]}, "
            f"displacement.shape[0]={self._displacement.shape[0]} "
        )

        node_shape = self._node_coords.shape

        sample_indices = np.arange(node_shape[0])
        np.random.shuffle(sample_indices)
        sample_indices = np.array_split(sample_indices, node_shape[0] // self.chunk_file_size)

        for file_i, indices in enumerate(sample_indices):
            file_path_group = f"{file_path}/data{file_i}.tfrecord"
            writer = tfrecord.TFRecordWriter(file_path_group)

            edge: np.ndarray = self._preprocess_edge(self._node_coords, indices)

            for i in range(len(indices)):
                context_data = {
                    "index": (indices[i], self.context_description["index"]),
                }
                feature_data: Dict[str, Tuple[np.ndarray, str]] = {
                    "node_coord": (self._node_coords[indices[i]], self.feature_description["node_coord"]),
                    "node_features": (self._node_features[indices[i]], self.feature_description["node_features"]),
                    "edges_indices": (edge[i], self.feature_description["edges_indices"]),
                    "shape_coeffs": (self._shape_coeffs[indices[i]], self.feature_description["shape_coeffs"]),
                    "theta_vals": (self._theta_vals[indices[i]], self.feature_description["theta_vals"]),
                    "labels": (self._displacement[indices[i]], self.feature_description["labels"]),
                }

                writer.write(context_data, feature_data)  # noqa

            writer.close()
            logger.info(f"File {file_i} written and closed")

    def _preprocess_edge(self, node_coords: np.ndarray, indices: np.array) -> np.ndarray:

        relative_positions = node_coords[indices, :, np.newaxis, :] - node_coords[indices, np.newaxis, :, :]
        relative_distance = np.sqrt(np.sum(np.square(relative_positions), axis=-1, keepdims=True))
        sorted_indices = np.argsort(relative_distance.squeeze(axis=-1), axis=-1)

        sorted_indices = self._random_select_nodes(sorted_indices[..., 1: 1001])  # remove the node itself

        return sorted_indices if self.gpu else sorted_indices

    def _random_select_nodes(self, indices: np.ndarray) -> np.ndarray:
        batch_size, rows, cols = indices.shape
        sections = [0, 20, 100, 200, 500, 1000]
        max_select_node = [20, 30, 30, 10, 10]
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
                    selected_indices[b, r, start_idx : start_idx + num_random_indices] = random_indices + range_start

        # Gather the selected indices from the original indices
        batch_indices = np.arange(batch_size)[:, None, None]
        row_indices = np.arange(rows)[None, :, None]
        selected_values = indices[batch_indices, row_indices, selected_indices]

        return selected_values

    def _data_stats(self) -> None:
        self._data_node_coords_stats()

        self._total_data_size()

    def _data_node_coords_stats(self) -> None:
        if os.path.exists(self.coord_max_norm_path) and os.path.exists(self.coord_min_norm_path):
            return

        node_coords = np.load(f"{self.raw_data_path}/real-node-coords.npy", mmap_mode="r").astype(np.float32)

        coord_max_norm_val = np.max(node_coords, axis=(0, 1))
        coord_min_norm_val = np.min(node_coords, axis=(0, 1))

        np.save(self.coord_max_norm_path, coord_max_norm_val)
        np.save(self.coord_min_norm_path, coord_min_norm_val)

    def _total_data_size(self) -> None:
        if os.path.exists(self.data_size_path):
            return

        node_coords = np.load(f"{self.raw_data_path}/real-node-coords.npy", mmap_mode="r").astype(np.float32)

        np.save(self.data_size_path, node_coords.shape[0])


class GraphSageTrainDataset(GraphSageDataset):
    """Data loader for graph-formatted input-output data with common, fixed topology."""

    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)

        # TODO: logic not reasonable here, need to be modified in the future
        if not os.listdir(self.tfrecord_path):
            preprocess_data = GraphSagePreprocessDataset(data_config, data_type)
            preprocess_data.preprocess()

        # node features/coord (used real node features)
        self.coord_max_norm_val = np.load(self.coord_max_norm_path).astype(np.float32)
        self.coord_min_norm_val = np.load(self.coord_min_norm_path).astype(np.float32)
        logger.info(
            f"{self.data_type} dataset max and min norm is " f"{self.coord_max_norm_val} {self.coord_min_norm_val}"
        )

        # summary statistics for displacement values calculated on training data
        self.displacement_mean = np.load(self.displacement_mean_path).astype(np.float32)
        self.displacement_std = np.load(self.displacement_std_path).astype(np.float32)

        # labels
        self.data_size = np.load(self.data_size_path).astype(np.int64).item()

        # file
        self.num_of_files = len(os.listdir(self.tfrecord_path))

        # init tfrecord loader config
        self.shuffle_queue_size = data_config.get("shuffle_queue_size", 5)
        self.compression_type = None

        # init transform data
        tfrecord_to_tensor_config = {
            "context_description": self.context_description,
            "feature_description": self.feature_description,
        }

        coord_max_min_config = {
            "node_coord": {
                max_val_name: torch.from_numpy(np.expand_dims(self.coord_max_norm_val, axis=0)),
                mim_val_name: torch.from_numpy(np.expand_dims(self.coord_min_norm_val, axis=0)),
            },
        }

        tensor_to_gpu_config = {"gpu": self.gpu, "cuda_core": self.cuda_core}

        convert_data_dim_config = {
            "theta_vals": -1,
            "shape_coeffs": -1
        }

        convert_model_input_config = {
            "labels": ["labels"]
        }

        self.transform = transforms.Compose([
            TFRecordToTensor(tfrecord_to_tensor_config),
            MaxMinNormalize(coord_max_min_config),
            ConvertDataDim(convert_data_dim_config),
            TensorToGPU(tensor_to_gpu_config),
            CovertToModelInputs(convert_model_input_config)
        ])

    def __len__(self):
        return self.data_size

    def __iter__(self) -> (Dict, torch.Tensor):
        shift, num_workers = 0, 0

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shift, num_workers = worker_info.id, worker_info.num_workers

        if num_workers > self.num_of_files:
            raise ValueError("the num of workers should be small or equal to num of files")

        if num_workers == 0:
            splits = {str(num): 1.0 for num in range(self.num_of_files)}
        else:
            splits = {str(num): 1.0 for num in range(self.num_of_files) if num % num_workers == shift}

        it = tfrecord.multi_tfrecord_loader(
            data_pattern=self.tfrecord_path+"/data{}.tfrecord",
            index_pattern=None,
            splits=splits,
            description=self.context_description,
            sequence_description=self.feature_description,
            compression_type=self.compression_type,
            infinite=False,
        )

        if self.shuffle_queue_size:
            it = tfrecord.shuffle_iterator(it, self.shuffle_queue_size)  # noqa

        it = map(self.transform, it)

        return it

    def get_displacement_mean(self) -> torch.tensor:
        _displacement_mean = torch.from_numpy(self.displacement_mean)
        return self.displacement_mean if not self.gpu else _displacement_mean.cuda()

    def get_displacement_std(self) -> torch.tensor:
        _displacement_std = torch.from_numpy(self.displacement_std)
        return _displacement_std if not self.gpu else _displacement_std.cuda()
