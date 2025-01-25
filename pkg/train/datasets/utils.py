import os
import sys
from typing import Dict

from pkg.utils.io import get_repo_path, load_yaml


def import_data_config(task_name: str, model_name: str, dataset_name: str = "") -> Dict:
    # generate root path
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = get_repo_path(cur_path)

    data_config = {}

    # fetch data config
    base_config = load_yaml(f"{repo_root_path}/task/{task_name}/{model_name}/config/data_config.yaml")
    data_config.update(base_config["task_data"])

    task_base = base_config["task_base"]
    data_config["task_name"] = task_base.get("task_name", task_name)
    # data_config["exp_name"] = task_base.get("exp_name", "default")
    data_config["model_name"] = task_base.get("model_name", model_name)

    data_config["repo_path"] = repo_root_path
    data_config["task_data_path"] = f"{repo_root_path}/pkg/data/{task_name}"
    data_config["task_path"] = f"{repo_root_path}/task/{task_name}/{model_name}"
    # data_config["gpu"] = base_config["task_base"].get("gpu", False)
    # data_config["exp_name"] = base_config["task_base"]["exp_name"]

    return data_config
