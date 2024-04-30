import os
import sys

from pkg.utils import io
from task.passive_lv_gnn_emul_11.train.datasets import LvDataset

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)
    data_config = {
        "task_data_path": f"{repo_root_path}/pkg/data/lvData",
    }

    data = LvDataset(data_config, "TRAIN")
