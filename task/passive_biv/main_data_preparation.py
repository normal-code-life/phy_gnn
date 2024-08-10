from common.constant import TRAIN_NAME, VALIDATION_NAME
from pkg.data.utils.dataset_generation import split_dataset_indices
from task.passive_biv.data.datasets import import_data_config
from task.passive_biv.data.datasets_preparation import \
    PassiveBiVPreparationDataset

if __name__ == "__main__":
    data_config = import_data_config()

    # generate sample indices
    sample_indices_dict = split_dataset_indices(
        data_config["sample_path"],
        data_config["train_split_ratio"],
    )

    # generate dataset
    for data_type in [TRAIN_NAME, VALIDATION_NAME]:
        data_config["sample_indices"] = sample_indices_dict[data_type]

        data = PassiveBiVPreparationDataset(data_config, data_type)

        data.prepare_dataset()
