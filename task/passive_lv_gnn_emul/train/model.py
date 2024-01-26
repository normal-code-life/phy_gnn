from pprint import pformat
from pkg.train.trainer.base_trainer import TrainerConfig, BaseTrainer
from pkg.train.model.base_model import ModelConfig, BaseModel
from pkg.utils.logging import init_logger
from pkg.utils import io
from task.passive_lv_gnn_emul.train.datasets import LvDataset
from common.constant import TRAIN_NAME, VALIDATION_NAME, TEST_NAME
import os
import sys


logger = init_logger("PassiveLvGNNEmul")


class Config(TrainerConfig):
    def __init__(self, root_path: str, config_path: str):
        super().__init__(root_path, config_path)
        logger.info(f"====== config init ====== \n{pformat(self.config)}")


class PassiveLvGNNEmulTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config)

        trainer_param = self.task_trainer["train_param"]
        train_param = self.task_train["model_custom_param"]

        logger.info(f"Data path: {self.task_data['task_data_path']}")
        logger.info(f'Message passing steps: {train_param["message_passing_steps"]}')
        logger.info(f'Num. shape coeffs: {train_param["n_shape_coeff"]}')
        logger.info(f'Training epochs: {trainer_param["step_param"]["epochs"]}')
        logger.info(f'Learning rate: {trainer_param["optimizer_param"]["learning_rate"]}')
        logger.info(f'Fixed LV geom: {trainer_param["fixed_geom"]}\n')

    def read_dataset(self):
        task_data = self.task_data

        train_data = LvDataset(task_data, TRAIN_NAME)
        logger.info(f"Number of train data points: {len(train_data)}")

        validation_data = LvDataset(task_data, VALIDATION_NAME)
        logger.info(f"Number of validation_data data points: {len(validation_data)}")

    def fit(self):
        self.read_dataset()


class PassiveLvGNNEmulModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__()


if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    task_dir = io.get_cur_abs_dir(cur_path)
    training_yaml_dir = f"{task_dir}/train_config.yaml"

    lv_config = Config(task_dir, training_yaml_dir)
    model = PassiveLvGNNEmulTrainer(lv_config)
    model.fit()
