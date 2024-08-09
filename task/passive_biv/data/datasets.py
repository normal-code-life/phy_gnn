import os
import sys
from typing import Dict

from pkg.train.datasets.base_datasets import MultiTFRecordDataset
from pkg.utils import io
from pkg.utils.io import load_yaml
from pkg.utils.logs import init_logger

logger = init_logger("PassiveBiV_Dataset")


class PassiveBiVDataset(MultiTFRecordDataset):
    """Passive BiV Dataset main class which including our basic attributes.

    This class is responsible for loading and processing data for a specific task,
    organized in a predefined directory structure. It supports reading from and saving
    to local paths, including data in tfrecord and npz formats. It also sets up the
    necessary paths for various data features and statistics.

    Parameters:
    ----------
    data_config : Dict
        A dictionary containing configuration information for the data. Expected keys include:
            - 'task_data_path': Base path for task-related data.
            - 'task_path': Path for task-specific files.
            - 'exp_name': (Optional) Name of the experiment.
            - 'default_padding_value': (Optional) Default padding value for data.

    data_type : str
        Type of data to be processed (e.g., 'train', 'test', 'validate').
    """

    def __init__(self, data_config: Dict, data_type: str) -> None:
        super().__init__(data_config, data_type)

        # node related features
        # === read data path
        self.inputs_data_path = f"{self.base_data_path}/record_inputs"

        # === save data path
        self.node_coord_stats_path = f"{self.stats_data_path}/node_coord_stats.npz"
        self.node_laplace_coord_stats_path = f"{self.stats_data_path}/node_laplace_stats.npz"
        self.fiber_and_sheet_stats_path = f"{self.stats_data_path}/fiber_and_sheet_stats.npz"

        logger.info(f"inputs_data_path is {self.inputs_data_path}")

        # global features
        # === read data path
        self.global_feature_data_path = f"{self.base_data_path}/record_global_feature.csv"
        self.shape_data_path = f"{self.base_data_path}/record_shape.csv"

        # === save data path
        self.mat_param_stats_path = f"{self.stats_data_path}/mat_param_stats.npz"
        self.pressure_stats_path = f"{self.stats_data_path}/pressure_stats.npz"
        self.shape_coeff_stats_path = f"{self.stats_data_path}/shape_coeff_stats.npz"

        logger.info(f"global_feature_data_path is {self.global_feature_data_path}")
        logger.info(f"shape_data_path is {self.shape_data_path}")

        # label
        # === read data path
        self.outputs_data_path = f"{self.base_data_path}/record_results"

        # === save data path
        self.displacement_stats_path = f"{self.stats_data_path}/displacement_stats.npz"
        self.stress_stats_path = f"{self.stats_data_path}/stress_stats.npz"

        logger.info(f"outputs_data_path is {self.outputs_data_path}")

        # others
        self.data_size_path = f"{self.stats_data_path}/" + "{}_data_size_value.npy".format(self.data_type)

        # features
        self.context_description = {
            "index": "int",
            "points": "int",
        }

        self.feature_description = {
            "node_coord": "float",
            "laplace_coord": "float",
            "fiber_and_sheet": "float",
            "edges_indices": "int",
            "shape_coeffs": "float",
            "mat_param": "float",
            "pressure": "float",
            "displacement": "float",
            "stress": "float",
        }

        self.labels = {"displacement", "stress"}


def import_data_config() -> Dict:
    # generate root path
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)

    # fetch data config
    config = load_yaml(f"{repo_root_path}/task/passive_biv/config/train_config.yaml")
    data_config = config["task_data"]
    data_config["task_data_path"] = f"{repo_root_path}/pkg/data/passive_biv"
    data_config["task_path"] = f"{repo_root_path}/task/passive_biv"
    data_config["sample_path"] = f"{data_config['task_data_path']}/record_inputs"
    data_config["gpu"] = config["task_base"]["gpu"]

    return data_config
