import os
import sys

from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.data.generate_data.dataset_generation import generate_dataset_indices
from pkg.utils import io
from pkg.utils.io import load_yaml
from task.passive_biv.data.datasets import PassiveBiVPreprocessDataset

if __name__ == "__main__":
    # generate root path
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)

    # fetch data config
    data_config = load_yaml(f"{repo_root_path}/task/passive_biv/config/train_config.yaml")["task_data"]
    data_config["task_data_path"] = f"{repo_root_path}/pkg/data/passive_biv"
    data_config["task_path"] = f"{repo_root_path}/task/passive_biv"
    data_config["sample_path"] = f"{data_config['task_data_path']}/record_inputs"
    data_config["gpu"] = False

    # generate sample indices
    sample_indices_dict = generate_dataset_indices(data_config)

    # generate dataset
    for data_type in [TRAIN_NAME, VALIDATION_NAME]:
        data_config["sample_indices"] = sample_indices_dict[data_type]

        data = PassiveBiVPreprocessDataset(data_config, data_type)

        data.preprocess()
