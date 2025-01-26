import argparse
from typing import List

from task.passive_lv.fe_heart_sage_v1.train.model import FEHeartSAGETrainer
from task.passive_lv.fe_heart_sage_v2.train.model import FEHeartSageV2Trainer
from task.passive_lv.fe_heart_sage_v3.train.model import FEHeartSageV3Trainer
from task.passive_lv.passive_lv_gnn_emul.train.model import PassiveLvGNNEmulTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Selection")

    parser.add_argument("--model_name", type=str, default="", help="model name")

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    if args[0].model_name == "fe_heart_sage_v1":
        model = FEHeartSAGETrainer()
    elif args[0].model_name == "fe_heart_sage_v2":
        model = FEHeartSageV2Trainer()
    elif args[0].model_name == "fe_heart_sage_v3":
        model = FEHeartSageV3Trainer()
    elif args[0].model_name == "passive_lv_gnn_emul":
        model = PassiveLvGNNEmulTrainer()
    else:
        raise ValueError("please pass the right model name")

    model.train()
