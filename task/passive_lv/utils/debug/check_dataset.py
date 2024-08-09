import os
import sys
from typing import Dict

from pkg.utils import io
from pkg.utils.io import load_yaml
from task.passive_lv.fe_heart_sage_v1.data.datasets import FEHeartSageV1Dataset


def import_data_config(model_name: str) -> Dict:
    # generate root path
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)

    # fetch data config
    base_config = load_yaml(f"{repo_root_path}/task/passive_lv/{model_name}/config/train_config.yaml")
    data_config = base_config["task_data"]
    data_config["task_data_path"] = f"{repo_root_path}/pkg/data/lvData"
    data_config["task_path"] = f"{repo_root_path}/task/passive_lv/{model_name}"
    data_config["gpu"] = base_config["task_base"]["gpu"]
    data_config["default_padding_value"] = -1
    data_config["chunk_size"] = 50

    return data_config


if __name__ == "__main__":
    config = import_data_config("fe_heart_sage_v1")

    data_preprocess = FEHeartSageV1Dataset(config, "train")
