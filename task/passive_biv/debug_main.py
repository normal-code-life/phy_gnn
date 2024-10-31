import argparse
import os
import sys
from typing import List

from pkg.utils import io
from task.passive_biv.fe_heart_sage_v1.train.model import FEHeartSAGETrainer
from task.passive_biv.fe_heart_sage_v2.train.model import FEHeartSageV2Trainer

# used for debug
if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])
    task_dir = io.get_repo_path(cur_path)
    sys.argv.extend(
        [
            "--repo_path",
            f"{task_dir}",
            "--task_name",
            "passive_biv",
            "--model_name",
            "fe_heart_sage_v1",
            "--config_name",
            "train_config",
        ]
    )

    parser = argparse.ArgumentParser(description="Model Selection")

    parser.add_argument("--model_name", type=str, default="", help="model name")

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    if args[0].model_name == "fe_heart_sage_v1":
        model = FEHeartSAGETrainer()
    elif args[0].model_name == "fe_heart_sage_v2":
        model = FEHeartSageV2Trainer()
    else:
        raise ValueError("please pass the right model name")

    model.train()
