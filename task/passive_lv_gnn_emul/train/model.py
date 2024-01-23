from pprint import pformat
from pkg.train.trainer.base_trainer import TrainerConfig, BaseTrainer
from pkg.utils.logging import init_logger
from pkg.utils import io
import os
import sys


logger = init_logger("PassiveLvGNNEmul")


class Config(TrainerConfig):
    def __init__(self, root_path: str, config_path: str):
        super().__init__(root_path, config_path)
        logger.info(f"====== config init ====== \n{pformat(self.config)}")


class PassiveLvGNNEmul(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config)

        trainer_param = self.task_trainer["train_param"]
        train_param = self.task_train["model_param"]

        logger.info(f"Data path: {config.task_data_path}")
        logger.info(f'Message passing steps (K): {train_param["K"]}')
        logger.info(f'Num. shape coeffs: {train_param["n_shape_coeff"]}')
        logger.info(f'Training epochs: {trainer_param["step_param"]["epochs"]}')
        logger.info(f'Learning rate: {trainer_param["optimizer_param"]["learning_rate"]}')
        logger.info(f'Fixed LV geom: {trainer_param["fixed_geom"]}\n')


if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    task_dir = io.get_cur_abs_dir(cur_path)
    training_yaml_dir = f"{task_dir}/train_config.yaml"

    config = Config(task_dir, training_yaml_dir)
    PassiveLvGNNEmul(config)
