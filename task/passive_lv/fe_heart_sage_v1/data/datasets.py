from typing import Dict

from pkg.train.datasets.base_datasets import BaseDataset
from pkg.utils.logs import init_logger
from pkg.utils.io import check_and_clean_path

logger = init_logger("FEHeartSage_Dataset")


class FEHeartSageV1Dataset(BaseDataset):
    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)

        self.default_padding_value = data_config.get("default_padding_value", -1)
        self.n_shape_coeff = data_config.get("n_shape_coeff", 2)
        self.sections = data_config["sections"]
        self.nodes_per_sections = data_config["nodes_per_sections"]
        self.overwrite_data = data_config.get("overwrite_data", True)

        self.raw_data_path = f"{self.base_data_path}/rawData/{self.data_type}"
        self.processed_data_path = f"{self.base_data_path}/processedData/{self.data_type}"
        self.topology_data_path = f"{self.base_data_path}/topologyData"
        self.stats_data_path = f"{self.base_data_path}/normalisationStatistics"
        self.fe_heart_sage_v1_data_path = f"{self.base_data_path}/fe_heart_sage_v1/{self.data_type}"

        if not check_and_clean_path(self.fe_heart_sage_v1_data_path, self.overwrite_data):
            return

        logger.info(f"base_data_path is {self.base_data_path}")
        logger.info(f"base_task_path is {self.base_task_path}")
        logger.info(f"raw_data_path is {self.raw_data_path}")
        logger.info(f"processed_data_path is {self.processed_data_path}")
        logger.info(f"topology_data_path is {self.topology_data_path}")
        logger.info(f"stats_data_path is {self.stats_data_path}")

        self.node_feature_path = f"{self.raw_data_path}/real-node-features.npy"
        self.node_coord_path = f"{self.processed_data_path}/real-node-coords.npy"
        self.edge_file_path = f"{self.processed_data_path}/node_neighbours_distance_{data_type}_{self.exp_name}.npy"
        self.theta_path = f"{self.processed_data_path}/global-features.npy"
        self.displacement_path = f"{self.processed_data_path}/real-node-displacement.npy"
        self.n_shape_coeff_path = f"{self.processed_data_path}/shape-coeffs.npy"


