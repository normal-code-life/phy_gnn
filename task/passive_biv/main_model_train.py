import argparse
from typing import List

from task.passive_biv.fe_heart_sim_sage.train.model import FEHeartSimSageTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Selection")

    parser.add_argument("--model_name", type=str, default="fe_heart_sim_sage", help="model name")

    args: (argparse.Namespace, List[str]) = parser.parse_known_args()

    if args[0].model_name == "fe_heart_sim_sage":
        model = FEHeartSimSageTrainer()
    else:
        raise ValueError("please pass the right model name")

    model.train()
