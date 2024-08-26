import time
from datetime import datetime
from typing import Dict

import torch

from pkg.train.callbacks.base_callback import CallBack
from pkg.utils.logs import init_logger

logger = init_logger("SCHEDULING_CALLBACK")


class SchedulingCallback(CallBack):
    """A callback to pause and resume training based on predefined time intervals.

    This callback checks whether the training should be paused during certain hours
    (e.g., peak hours during weekdays) and handles moving the model between CPU
    and GPU to manage GPU resources efficiently.

    Methods:
    --------
    on_epoch_begin(epoch, **kwargs):
        Checks if training should be paused and handles the transfer of the model
        between CPU and GPU.

    is_within_pause_time():
        Determines if the current time is within the designated pause period.
    """

    def __init__(self, task_base_param: Dict, param: Dict) -> None:
        super(SchedulingCallback, self).__init__(task_base_param, param)

        self.dive_in_sleeping_time = False

    def on_epoch_begin(self, epoch, **kwargs) -> None:
        """Called at the beginning of each epoch. This method pauses training during specified hours.

        During the sleeping time, it will move the model to CPU to release GPU resources. It
        resumes training by moving the model back to GPU.

        Parameters:
        -----------
        epoch : int
            The current epoch number.

        kwargs : dict
            Additional keyword arguments (not used in this method).
        """
        while self.is_within_pause_time():
            logger.info(f"{datetime.now()} Training paused. Releasing GPU resources if necessary...")

            if self.use_gpu:
                self.model.cpu()  # Move model to CPU to free GPU memory

                torch.cuda.empty_cache()  # Clear GPU memory

            time.sleep(1)  # Check every minute if training can resume

            self.dive_in_sleeping_time = True

        # When resuming, move the model back to the GPU
        if self.use_gpu and self.dive_in_sleeping_time:
            self.model.cuda()

    @staticmethod
    def is_within_pause_time() -> bool:
        """Determines if the current time is within the specified pause period.

        The pause period is defined as weekdays (Monday to Friday) between 10 AM
        and 6 PM, during which training is paused to manage GPU usage.

        Returns:
        --------
        bool : True if the current time is within the pause period, otherwise False.
        """
        current_time = datetime.now()
        current_hour = current_time.hour
        current_weekday = current_time.weekday()

        # Weekday is Monday=0 to Sunday=6, we pause only on weekdays (0-4) between 10 - 18 hours
        if 0 <= current_weekday <= 5:
            if 10 <= current_hour < 18:
                return True

        return False
