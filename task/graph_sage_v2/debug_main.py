import os
import sys

from pkg.utils import io
from task.graph_sage_v2.train.model import GraphSAGETrainer

# used for debug

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])
    task_dir = io.get_repo_path(cur_path)
    sys.argv.extend(
        ["--repo_path", f"{task_dir}", "--task_name", "graph_sage_v2", "--config_name=train_config_lv_data"]
    )

    model = GraphSAGETrainer()
    model.train()
