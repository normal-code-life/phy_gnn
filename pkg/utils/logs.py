import logging


def init_logger(logger_name: str = "gnn", debug_level: str = "INFO") -> logging:
    """
    Initialize and configure a logger.

    Args:
        logger_name (str, optional): The name of the logger. Default is "gnn".
        debug_level (str, optional): The debug level that controls the verbosity of log records.
        Default is DebugLevel.Info.

    Returns:
        logging.Logger: The configured logger.

    Note:
        1. If the specified logger name already exists, the existing logger is returned.
        2. The debug_level parameter accepts values "DEBUG", "INFO", "WARNING", "ERROR".
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(debug_level)

    handler = logging.StreamHandler()
    handler.setLevel(debug_level)

    formatter = logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False

    return logger
