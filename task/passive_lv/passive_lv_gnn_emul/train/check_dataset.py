import os
import sys

from pkg.utils import io
from task.passive_lv.passive_lv_gnn_emul.train.datasets import LvDataset

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)
    data_config = {
        "repo_path": repo_root_path,
        "task_data_path": f"{repo_root_path}/pkg/data/lvData",
        "task_path": f"{repo_root_path}/task/passive_lv/passive_lv_gnn_emul",
        "sub_data_name": "beam_data",
        "gpu": False,
        "model_name": "passive_lv_gnn_emul",
        "task_name": "model_train",
        "exp_name": "v1",
        "n_shape_coeff": 11,
    }

    data = LvDataset(data_config, "TRAIN")
