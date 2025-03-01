import os
from typing import Dict

import numpy as np

from common.constant import TRAIN_NAME, VALIDATION_NAME


def split_dataset_indices(sample_path: str, train_split_ratio: float) -> Dict[str, np.ndarray]:
    """Splits a dataset into training and validation sets by randomly shuffling and dividing the indices.

    This function takes a directory of samples and splits them into training and validation sets
    based on the provided train_split_ratio. It generates random indices for the split to ensure
    the data distribution is random between sets.

    Parameters:
    ----------
    sample_path : str
        Path to the directory containing the dataset samples. The function will count
        the total number of files in this directory to determine dataset size.
    train_split_ratio : float
        Ratio of samples to use for training, must be between 0 and 1. For example,
        0.8 means 80% of samples will be used for training and 20% for validation.

    Returns:
    -------
    Dict[str, np.ndarray]
        A dictionary containing two keys:
        - TRAIN_NAME: np.ndarray of indices for training samples
        - VALIDATION_NAME: np.ndarray of indices for validation samples

    Example:
    -------
    >>> split = split_dataset_indices("/path/to/samples", 0.8)
    >>> train_indices = split[TRAIN_NAME]
    >>> validation_indices = split[VALIDATION_NAME]
    """
    # Get total number of samples from directory
    total_sample_size = len(os.listdir(sample_path))

    # Calculate number of training samples based on split ratio
    train_sample_size: int = int(total_sample_size * train_split_ratio)

    # Generate shuffled indices for the entire dataset
    sample_indices: np.ndarray = np.arange(total_sample_size)
    np.random.shuffle(sample_indices)

    # Split indices into training and validation sets
    train_indices: np.ndarray = sample_indices[:train_sample_size]
    validation_indices: np.ndarray = sample_indices[train_sample_size:]

    return {TRAIN_NAME: train_indices, VALIDATION_NAME: validation_indices}
