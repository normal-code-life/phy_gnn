import os
import sys

from torch.utils.data import DataLoader

from pkg.utils import io
from task.graph_sage_v2.data.datasets import GraphSageTrainDataset

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

    train_data = GraphSageTrainDataset(data_config, "train")

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=data_config.get("batch_size", 1),
        num_workers=data_config.get("num_workers", 3),
        prefetch_factor=data_config.get("prefetch_factor", None),
    )

    s = 0
    for i in range(5):
        for context, labels in train_data_loader:
            s += 1
            print(s, context["index"])
