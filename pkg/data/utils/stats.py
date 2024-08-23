import logging
from typing import Set, Union

import numpy as np

from common.constant import (
    MAX_VAL,
    MEAN_VAL,
    MEDIAN_VAL,
    MIN_VAL,
    PERC_10_VAL,
    PERC_25_VAL,
    PERC_75_VAL,
    PERC_90_VAL,
    STD_VAL
)


def stats_analysis(
    feature_name: str,
    value_set: np.ndarray,
    axis: Union[Set[int], int],
    save_path: str,
    logger: logging,
    write_to_path: bool = False,
) -> None:
    """
    Analyze statistical properties of a given numpy array and save the results.

    Parameters:
    ----------
    feature_name : str
        The name of the feature being analyzed.
    value_set : np.ndarray
        The numpy array containing the values to analyze.
    axis : List[int]
        The axis or axes along which to compute the statistics. If None, compute over the entire array.
    save_path : str
        The path to save the resulting statistics as a .npz file.
    logger : logging.Logger
        The logger used to log the statistical analysis results.
    write_to_path : bool
        decide whether to write to path

    Returns:
    ----------
    None
    """
    # Compute the statistical values
    stats_val = {
        MAX_VAL: np.max(value_set, axis=axis),
        MIN_VAL: np.min(value_set, axis=axis),
        MEAN_VAL: np.mean(value_set, axis=axis),
        STD_VAL: np.std(value_set, axis=axis),
        MEDIAN_VAL: np.median(value_set, axis=axis),
        PERC_10_VAL: np.percentile(value_set, [10], axis=axis),
        PERC_25_VAL: np.percentile(value_set, [25], axis=axis),
        PERC_75_VAL: np.percentile(value_set, [75], axis=axis),
        PERC_90_VAL: np.percentile(value_set, [90], axis=axis),
    }

    # Log the statistics
    logger.info(f"stats feature name: {feature_name}, shape: {value_set.shape}, stats data: ")
    for key in stats_val:
        logger.info(f"{feature_name}: {key}, {stats_val[key]}")

    # Save the statistics to a .npz file
    if write_to_path:
        np.savez(save_path, **stats_val)
