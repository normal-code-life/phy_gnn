import os
import sys

from pkg.utils import io
from task.passive_lv_gnn_emul.train.model import PassiveLvGNNEmulTrainer

# used for debug

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])
    task_dir = io.get_repo_path(cur_path)
    sys.argv.extend(
        ["--repo_path", f"{task_dir}", "--task_name", "passive_lv_gnn_emul", "--config_name=train_config"]
    )

    model = PassiveLvGNNEmulTrainer()
    model.train()
