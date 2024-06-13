import os
import sys

from pkg.utils import io
from task.graph_sage.data.datasets import GraphSageDataset

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)
    data_config = {
        "task_data_path": f"{repo_root_path}/pkg/data/lvData",
        "task_path": f"{repo_root_path}/task/graph_sage",
        "gpu": False,
        "n_shape_coeff": 32,
        "edge_indices_generate_method": 2,
        "default_padding_value": -1,
    }

    data = GraphSageDataset(data_config, "TRAIN")
