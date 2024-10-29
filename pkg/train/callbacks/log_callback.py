import os
import time
from typing import Dict

from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logs import init_logger

logger = init_logger("LOGS_CALLBACK")


class LogCallback(CallBack):
    """A custom callback for logging training metrics and optionally saving configurations and code.

    Parameters:
    -----------
    task_base_param : Dict
        Dictionary containing base parameters related to the task, such as config_path.

    param : Dict
        Dictionary of additional parameters, such as update frequency and flags for saving config and code.

    Attributes:
    -----------
    config_path : str
        Path to the configuration file.

    start_time : float
        Time when the training begins, used to measure the elapsed time.

    update_freq : str or int
        Frequency at which the logs should be updated. Can be 'epoch' or an integer representing the number of epochs.

    save_config : bool
        Flag indicating whether to save the configuration file.

    save_task_code : bool
        Flag indicating whether to save the task code.
    """

    def __init__(self, task_base_param: Dict, param: Dict) -> None:
        super(LogCallback, self).__init__(task_base_param, param)

        self.config_path = task_base_param["config_path"]

        self.start_time = time.time()

        self.update_freq = param.get("update_freq", "epoch")

        self.save_config = param.get("save_config", False)

        self.save_task_code = param.get("save_task_code", False)

    def on_train_begin(self, **kwargs):
        """Called at the beginning of training. Optionally saves the config file or code based on parameters."""
        if self.save_config:
            cmd = f"cp {self.config_path} {self.log_dir}/"
            logger.info(f"execute {cmd}")
            os.system(cmd)

        if self.save_task_code:
            cmd = f"cp -r {self.task_dir} {self.log_dir}/code/"
            logger.info(f"execute {cmd}")
            os.system(cmd)

    def on_evaluation_end(self, epoch, **kwargs):
        """Called at the end of evaluation end.

        Parameters:
        -----------
        epoch : int
            The current epoch number.

        kwargs : dict
            Dictionary containing training and validation metrics.
        """
        self._print_log(epoch, **kwargs)

    def on_epoch_end(self, epoch, **kwargs):
        """Called at the end of each epoch.

        Parameters:
        -----------
        epoch : int
            The current epoch number.

        kwargs : dict
            Dictionary containing training and validation metrics.
        """
        self._print_log(epoch, **kwargs)

    def _print_log(self, epoch, **kwargs):
        """Logs training and validation metrics.

        Parameters:
        -----------
        epoch : int
            The current epoch number.

        kwargs : dict
            Dictionary containing training and validation metrics.
        """
        train_metrics = dict()
        val_metrics = dict()

        if "train_metrics" in kwargs:
            train_metrics = kwargs["train_metrics"]

        if "val_metrics" in kwargs:
            val_metrics = kwargs["val_metrics"]

        if self.update_freq == "epoch" or epoch % self.update_freq == 0:
            train_logs = {
                **{name: val for name, val in train_metrics.items()},
                **{name: val for name, val in val_metrics.items()},
            }

            # Format the log message
            msg = " - ".join(f"{name}: {round(val, 5)}" for name, val in train_logs.items())

            # Log the message, including the current epoch and elapsed time
            logger.info(f"metrics: {epoch} - {round(time.time() - self.start_time, 2)}s - {msg}")
