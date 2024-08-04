import os
import sys

from pkg.utils import io
from task.passive_biv.train.model import PassiveBiVTrainer

# used for debug

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])
    task_dir = io.get_repo_path(cur_path)
    sys.argv.extend(["--repo_path", f"{task_dir}", "--task_name", "passive_biv", "--config_name=train_config"])

    model = PassiveBiVTrainer()
    model.train()
