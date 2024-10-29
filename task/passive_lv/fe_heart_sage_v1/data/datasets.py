import os
import sys
from typing import Dict

from pkg.train.datasets.base_datasets import BaseAbstractDataset
from pkg.utils.io import get_repo_path, load_yaml
from task.passive_lv.fe_heart_sage_v1.data import logger


class FEHeartSageV1Dataset(BaseAbstractDataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)
        self.default_padding_value = data_config.get("default_padding_value", -1)
        self.n_shape_coeff = data_config.get("n_shape_coeff", 2)
        self.sections = data_config["sections"]
        self.nodes_per_sections = data_config["nodes_per_sections"]
        self.overwrite_data = data_config.get("overwrite_data", True)

        # original base data path
        self.raw_data_path = f"{self.base_data_path}/rawData/{data_type}"
        self.processed_data_path = f"{self.base_data_path}/processedData/{data_type}"
        self.topology_data_path = f"{self.base_data_path}/topologyData"

        logger.info(f"raw_data_path is {self.raw_data_path}")
        logger.info(f"processed_data_path is {self.processed_data_path}")
        logger.info(f"topology_data_path is {self.topology_data_path}")

        # original data file path
        self.node_feature_original_path = f"{self.raw_data_path}/real-node-features.npy"
        self.node_coord_original_path = f"{self.processed_data_path}/real-node-coords.npy"
        self.theta_original_path = f"{self.processed_data_path}/global-features.npy"
        self.displacement_original_path = f"{self.processed_data_path}/real-node-displacement.npy"
        self.shape_coeff_original_path = f"{self.processed_data_path}/shape-coeffs.npy"

        # clean data destination path
        self.dataset_path = f"{self.base_data_path}/datasets/{self.model_name}/{self.data_type}"

        self.node_feature_path = f"{self.dataset_path}/node_features.npy"
        self.node_coord_path = f"{self.dataset_path}/node_coords.npy"
        self.edge_file_path = f"{self.dataset_path}/node_neighbours_{self.data_type}_{self.exp_name}.npy"
        self.theta_path = f"{self.dataset_path}/global_features.npy"
        self.displacement_path = f"{self.dataset_path}/node_displacement.npy"
        self.shape_coeff_path = f"{self.dataset_path}/shape_coeffs.npy"

        logger.info(f"we have overwrite the dataset_path: {self.dataset_path}")
        logger.info(f"node_feature_path is {self.node_feature_path}")
        logger.info(f"node_coord_path is {self.node_coord_path}")
        logger.info(f"edge_file_path is {self.edge_file_path}")
        logger.info(f"theta_path is {self.theta_path}")
        logger.info(f"displacement_path is {self.displacement_path}")
        logger.info(f"shape_coeff_path is {self.shape_coeff_path}")

        # stats path
        self.node_coord_max_path = f"{self.stats_data_path}/node_coords_max.npy"
        self.node_coord_min_path = f"{self.stats_data_path}/node_coords_min.npy"

        self.displacement_mean_path = f"{self.base_data_path}/normalisationStatistics/real-node-displacement-mean.npy"
        self.displacement_std_path = f"{self.base_data_path}/normalisationStatistics/real-node-displacement-std.npy"
