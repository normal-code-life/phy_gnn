"""I/O Utility Module

This module provides common file and directory operations including:
- Getting absolute paths and repository root paths
- Loading YAML configuration files
- Managing directory creation and cleanup

The module is designed for:
- Consistent path handling across the codebase
- Safe directory management with overwrite protection
- Configuration file loading
- Logging of file operations
"""

import os
import shutil
from typing import Dict

import yaml

from pkg.utils.logs import init_logger

logger = init_logger("IO")


def get_cur_abs_dir(path) -> str:
    """Get the absolute directory path of a file.

    Args:
        path: Path to get directory from

    Returns:
        Absolute directory path
    """
    abs_dir = os.path.dirname(path)
    return abs_dir


def get_repo_path(path: str) -> str:
    """Get the root repository path by looking for .git directory.

    Traverses up directory tree until finding .git folder.

    Args:
        path: Starting path to search from

    Returns:
        Path to repository root
    """
    # Get the directory of the current script
    script_directory = os.path.dirname(path)

    # Traverse upwards until a directory containing the .git folder is found, indicating the root of the repo
    while script_directory and not os.path.exists(os.path.join(script_directory, ".git")):
        script_directory = os.path.dirname(script_directory)

    return script_directory


def load_yaml(path: str) -> Dict:
    """Load and parse a YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary containing parsed YAML data
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def check_and_clean_path(path: str, overwrite: bool) -> bool:
    """Check if a directory exists and optionally clean it.

    Args:
        path: Directory path to check/clean
        overwrite: Whether to delete existing directory contents

    Returns:
        True if directory is ready for use, False if exists and overwrite=False
    """
    if os.path.exists(path):
        size = len(os.listdir(path))
        logger.info(f"directory of {path} size: {size}")
        if size > 0:
            if overwrite:
                shutil.rmtree(path)
                os.makedirs(path)
                logger.info("clean directory file to 0")
                return True
            else:
                return False
        else:
            return True
    else:
        os.makedirs(path)
        return True
