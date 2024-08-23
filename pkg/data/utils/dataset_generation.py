import os
from typing import Dict

import numpy as np

from common.constant import TRAIN_NAME, VALIDATION_NAME


def split_dataset_indices(sample_path: str, train_split_ratio: float) -> Dict[str, np.ndarray]:
    """Splits the dataset into training and validation indices according to the specified ratios and chunk sizes.

    Parameters:
    ----------
    data_config: Dict
        A dictionary containing configuration parameters.

    sample_path: str
        Path to the directory containing the dataset samples.
    train_split_ratio: float
        Ratio of the dataset to be used for training.

    Returns:
    ----------
    Dict[str, List[np.ndarray]]: A dictionary containing the indices for the training and validation datasets.
        - 'TRAIN_NAME': A list of NumPy arrays, each array containing indices of the training samples.
        - 'VALIDATION_NAME': A list of NumPy arrays, each array containing indices of the validation samples.
    """
    # check the record inputs size and split train and test dataset
    total_sample_size = len(os.listdir(sample_path))

    train_sample_size: int = int(total_sample_size * train_split_ratio)

    sample_indices: np.ndarray = np.arange(total_sample_size)
    np.random.shuffle(sample_indices)

    train_indices: np.ndarray = sample_indices[:train_sample_size]

    validation_indices: np.ndarray = sample_indices[train_sample_size:]

    return {TRAIN_NAME: train_indices, VALIDATION_NAME: validation_indices}
