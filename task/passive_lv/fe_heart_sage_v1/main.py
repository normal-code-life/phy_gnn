import yaml

from task.passive_lv.fe_heart_sage_v1.train.model import GraphSAGETrainer

if __name__ == "__main__":
    # Load training config
    config_path = "task/passive_lv/fe_heart_sage_v3/config/train_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize and train model with config
    model = GraphSAGETrainer(config)
    model.train()
