import argparse
import os
import sys
from typing import List

from pkg.utils import io
from task.passive_lv.fe_heart_sage_v1.train.model import FEHeartSAGETrainer
from task.passive_lv.fe_heart_sage_v2.train.model import FEHeartSageV2Trainer
from task.passive_lv.passive_lv_gnn_emul.train.model import PassiveLvGNNEmulTrainer

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
            # "fe_heart_sage_v2",
            "passive_lv_gnn_emul",
            "--config_name",
            # "train_config",
            "train_config_lv_data",
            "--task_type",
            "model_train",
        ]
    )

    parser = argparse.ArgumentParser(description="Model Selection")

    parser.add_argument("--model_name", type=str, default="", help="model name")

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    if args[0].model_name == "fe_heart_sage_v1":
        model = FEHeartSAGETrainer()
    elif args[0].model_name == "fe_heart_sage_v2":
        model = FEHeartSageV2Trainer()
    elif args[0].model_name == "passive_lv_gnn_emul":
        model = PassiveLvGNNEmulTrainer()
    else:
        raise ValueError("please pass the right model name")

    model.train()
