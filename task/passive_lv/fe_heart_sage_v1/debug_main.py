import os
import sys

from pkg.utils import io
from task.passive_lv.fe_heart_sage_v1.train.model import FEHeartSAGETrainer  # noqa
from task.passive_lv.fe_heart_sage_v2.train.model import FEHeartSageV2Trainer  # noqa

# used for debug
if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])
    task_dir = io.get_repo_path(cur_path)
    sys.argv.extend(
        [
            "--repo_path",
            f"{task_dir}",
            "--task_name",
            "passive_lv",
            "--model_name",
            "fe_heart_sage_v2",
            "--config_name",
            "train_config",
        ]
    )

    model = FEHeartSageV2Trainer()
    model.train()
