import os
import sys

from pkg.utils import io
from task.graph_sage_v2.data.datasets import GraphSagePreprocessDataset

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)
    data_config = {
        "task_data_path": f"{repo_root_path}/pkg/data/lvData",
        "task_path": f"{repo_root_path}/task/graph_sage_v2",
        "gpu": False,
        "n_shape_coeff": 32,
        "default_padding_value": -1,
        "exp_name": "v3_1",
        "num_of_files": 5,
    }

    for data_type in ["train", "validation"]:
        data = GraphSagePreprocessDataset(data_config, data_type)

        data.preprocess()
