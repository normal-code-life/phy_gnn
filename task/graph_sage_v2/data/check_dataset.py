import os
import sys

from pkg.utils import io
from task.graph_sage_v2.data.datasets import GraphSagePreprocessDataset, GraphSageTrainDataset


if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)
    data_config = {
        "task_data_path": f"{repo_root_path}/pkg/data/lvData",
        "task_path": f"{repo_root_path}/task/graph_sage",
        "gpu": False,
        "n_shape_coeff": 32,
        "edge_indices_generate_method": 0,
        "default_padding_value": -1,
        "exp_name": "10",
        "chunk_size": 20,
        "shuffle_queue_size": 3,
    }

    train_data = GraphSageTrainDataset(data_config, "validation")

    s = 0

    for context, labels in train_data:
        s += 1
        print(s)
        print(context, labels)
