import os
from typing import Dict, Tuple, List, Any

import numpy as np
import torch
import torch.utils.data
from pkg.train.datasets.base_datasets import BaseIterableDataset
from pkg.utils.logging import init_logger
import tfrecord

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
        self.chunk_size = data_config.get("chunk_size", 1)

        if not os.path.isdir(base_data_path):
            raise NotADirectoryError(f"No directory at: {base_data_path}")
        else:
            logger.info(f"base_data_path is {base_data_path}")

        self.raw_data_path = f"{base_data_path}/rawData/{self.data_type}"
        self.processed_data_path = f"{base_data_path}/processedData/{self.data_type}"
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

        self.tfrecord_path = f"{self.processed_data_path}/{self.data_type}_data.tfrecord"

        # others
        self.data_size_path = f"{self.processed_data_path}/{self.data_type}_data_size.npy"


class GraphSagePreprocessDataset(GraphSageDataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super(GraphSagePreprocessDataset, self).__init__(data_config, data_type)

    def preprocess(self):
        self._preprocess_data()
        self._data_stats()

    def _preprocess_data(self):
        file_path = self.tfrecord_path

        if os.path.exists(file_path):
            return

        # node features/coord (used real node features)
        # === coord max min calculation
        self._node_coords = np.load(self.real_node_coord_path).astype(np.float32)

        self.coord_max_norm_val = np.load(self.coord_max_norm_path).astype(np.float32)
        self.coord_min_norm_val = np.load(self.coord_min_norm_path).astype(np.float32)
        logger.info(
            f"{self.data_type} dataset max and min norm is " f"{self.coord_max_norm_val} {self.coord_min_norm_val}"
        )

        self._node_features = np.load(self.real_node_features_path).astype(np.float32)

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

        assert (self._node_coords.shape[1] ==
                self._node_features.shape[1] ==
                self._displacement.shape[1]
                ), (
            f"Variables are not equal: "
            f"node_coords.shape[0]={self._node_coords.shape[0]}, "
            f"node_features.shape[0]={self._node_features.shape[0]}, "
            f"displacement.shape[0]={self._displacement.shape[0]} "
        )

        node_shape = self._node_coords.shape

        writer = tfrecord.TFRecordWriter(file_path)

        array = np.arange(node_shape[0])
        np.random.shuffle(array)

        for i in array:
            edge: np.ndarray = self._preprocess_edge(self._node_coords, i)

            seq_data: Dict[str, Tuple[np.ndarray, str]] = {
                "node_coord": (self._node_coords[i], "float"),
                "node_features": (self._node_features[i], "float"),
                "edges_indices": (edge, "float"),
                "shape_coeffs": (self._shape_coeffs[i], "float"),
                "theta_vals": (self._theta_vals[i], "float"),
                "labels": (self._displacement[i], "float"),
            }

            writer.write({}, seq_data)  # noqa
            logger.info(f"write down the seq_data for {i}")

        writer.close()

    def _preprocess_edge(self, node_coords: np.ndarray, i: int) -> np.ndarray:

        if self.gpu:
            node_coords = torch.tensor(node_coords, device="cuda")
        else:
            node_coords = torch.tensor(node_coords)

        relative_positions = node_coords[i, :, None, :] - node_coords[i, None, :, :]

        relative_distance = torch.sqrt(torch.sum(torch.square(relative_positions), dim=-1, keepdim=True))

        sorted_indices = torch.argsort(relative_distance.squeeze(-1), dim=-1)

        sorted_indices = sorted_indices[..., 1:].to(torch.int32)  # remove the node itself

        return sorted_indices.cpu().numpy() if self.gpu else sorted_indices.numpy()

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

        if not os.path.exists(self.tfrecord_path):
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

        self.data_path = f"{self.tfrecord_path}"
        self.index_path = None
        self.context_description = None
        self.compression_type = None
        self.sequence_description = {
            "node_coord": "float",
            "node_features": "float",
            "edges_indices": "float",
            "shape_coeffs": "float",
            "theta_vals": "float",
            "labels": "float",
        }
        self.shuffle_queue_size = data_config.get("shuffle_queue_size", 5)

    def __len__(self):
        return self.data_size

    def __iter__(self) -> (Dict, torch.Tensor):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None

        it = tfrecord.tfrecord_loader(data_path=self.data_path,
                                      index_path=self.index_path,
                                      description=self.context_description,
                                      shard=shard,
                                      sequence_description=self.sequence_description,
                                      compression_type=self.compression_type)

        if self.shuffle_queue_size:
            it = tfrecord.shuffle_iterator(it, self.shuffle_queue_size)

        print(it)

        return it

        # context_feature, seq_feature = it
        #
        # node_coord = self._normal_max_min_transform(
        #     torch.tensor(np.array(seq_feature["node_coord"]), dtype=torch.float32),
        #     self.coord_max_norm_val,
        #     self.coord_min_norm_val,
        # )
        #
        # node_features = torch.tensor(np.array(seq_feature["node_features"]), dtype=torch.float32)
        # edges_indices = torch.tensor(np.array(seq_feature["edges_indices"]), dtype=torch.int64)
        # theta_vals = torch.tensor(np.array(seq_feature["theta_vals"]), dtype=torch.float32).squeeze(dim=-1)
        # shape_coeffs = torch.tensor(np.array(seq_feature["shape_coeffs"]), dtype=torch.float32).squeeze(dim=-1)
        # labels = torch.tensor(np.array(seq_feature["labels"]), dtype=torch.float32)
        #
        # if self.gpu:
        #     node_coord = node_coord.cuda()
        #     node_features = node_features.cuda()
        #     edges_indices = edges_indices.cuda()
        #     theta_vals = theta_vals.cuda()
        #     shape_coeffs = shape_coeffs.cuda()
        #
        #     labels = labels.cuda()
        #
        # yield {
        #     "node_coord": node_coord,
        #     "node_features": node_features,
        #     "edges_indices": edges_indices,
        #     "shape_coeffs": shape_coeffs,
        #     "theta_vals": theta_vals,
        # }, labels

    def _normal_max_min_transform(
            self, array: torch.Tensor, max_norm_val: np.float32, min_norm_val: np.float32
    ) -> torch.Tensor:
        max_val = torch.from_numpy(np.expand_dims(max_norm_val, axis=(0)))
        min_val = torch.from_numpy(np.expand_dims(min_norm_val, axis=(0)))

        return (array - min_val) / (max_val - min_val)

    def get_displacement_mean(self) -> torch.tensor:
        _displacement_mean = torch.from_numpy(self.displacement_mean)
        return self.displacement_mean if not self.gpu else _displacement_mean.cuda()

    def get_displacement_std(self) -> torch.tensor:
        _displacement_std = torch.from_numpy(self.displacement_std)
        return _displacement_std if not self.gpu else _displacement_std.cuda()
