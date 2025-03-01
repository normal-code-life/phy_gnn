"""Statistical Analysis Module

This module provides functionality for computing and analyzing statistical properties
of numerical data arrays. It calculates common statistical measures like mean, median,
standard deviation and various percentiles.

The module is designed for:
- Computing descriptive statistics on numpy arrays
- Handling multi-dimensional data with flexible axis selection
- Logging statistical results
- Optionally saving results to disk
"""

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
    PERC_95_VAL,
    PERC_99_VAL,
    STD_VAL
)


def stats_analysis(
    feature_name: str,
    value_set: np.ndarray,
    axis: Union[Set[int], int],
    save_path: str,
    logger: logging.Logger,
    write_to_path: bool = False,
) -> None:
    """Computes and logs statistical measures for a numerical array.

    Calculates common statistical properties including:
    - Min, max, mean, median
    - Standard deviation
    - Various percentiles (10th, 25th, 75th, 90th, 95th, 99th)

    Parameters
    ----------
    feature_name : str
        Name identifier for the feature being analyzed
    value_set : np.ndarray
        Array of numerical values to analyze
    axis : Union[Set[int], int]
        Axis or axes along which to compute statistics
    save_path : str
        Path where statistics will be saved if write_to_path is True
    logger : logging.Logger
        Logger instance for outputting results
    write_to_path : bool, optional
        Whether to save results to disk, by default False

    Returns
    -------
    None
        Results are logged and optionally saved to disk
    """
    # Calculate statistical measures
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
        PERC_95_VAL: np.percentile(value_set, [95], axis=axis),
        PERC_99_VAL: np.percentile(value_set, [99], axis=axis),
    }

    # Log results
    logger.info(f"Stats for feature '{feature_name}' with shape {value_set.shape}:")
    for key, value in stats_val.items():
        logger.info(f"{key}: {value}")

    # Optionally save to disk
    if write_to_path:
        np.savez(save_path, **stats_val)
