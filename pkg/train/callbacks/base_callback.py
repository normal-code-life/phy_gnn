"""export from keras."""
import abc
from typing import Dict, List, Optional

from torch import nn
from torch.optim import Optimizer
from pkg.utils.logs import init_logger

logger = init_logger("CALLBACK")


class CallBack(abc.ABC):
    model: nn.Module
    optimizer: Optimizer

    def __init__(self, task_base_param: Dict, logs_param: Dict):
        self.params: Dict = dict()

        self.task_dir = task_base_param["task_path"]

        if "log_dir" not in logs_param:
            self.log_dir = task_base_param["logs_base_path"]
        else:
            self.log_dir = logs_param["log_dir"]

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def on_batch_begin(self, batch, **kwargs):
        """A backwards compatibility alias for `on_train_batch_begin`."""
        return

    def on_batch_end(self, batch, **kwargs):
        """A backwards compatibility alias for `on_train_batch_end`."""
        return

    def on_epoch_begin(self, epoch, **kwargs):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        return

    def on_epoch_end(self, epoch, **kwargs):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        return

    def on_train_batch_begin(self, batch, **kwargs):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently, no data is passed to this argument for this
              method but that may change in the future.
        """
        # For backwards compatibility.
        return

    def on_train_batch_end(self, batch, **kwargs):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        # For backwards compatibility.
        return

    def on_validation_batch_begin(self, batch, **kwargs):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently, no data is passed to this argument for this
              method but that may change in the future.
        """
        return

    def on_validation_batch_end(self, batch, **kwargs):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        return

    def on_train_begin(self, **kwargs):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently, no data is passed to this argument for this
              method but that may change in the future.
        """
        return

    def on_train_end(self, **kwargs):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently, the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """
        return

    def on_validation_begin(self, **kwargs):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently, no data is passed to this argument for this
              method but that may change in the future.
        """
        return

    def on_validation_end(self, **kwargs):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently, the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """
        return


class CallbackList(object):
    """Container abstracting a list of callbacks."""

    def __init__(self, callbacks: Optional[List[CallBack]], model: nn.Module, optimizer: Optimizer):
        self.callbacks = callbacks

        self.set_model(model)
        self.set_optimizer(optimizer)

    def set_model(self, model: nn.Module):
        for callback in self.callbacks:
            callback.set_model(model)

    def set_optimizer(self, optimizer: Optimizer):
        for callback in self.callbacks:
            callback.set_optimizer(optimizer)

    def append(self, callback):
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch, **kwargs):
        """Calls the `on_epoch_begin` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch, **kwargs):
        """Calls the `on_epoch_end` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
        """
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, **kwargs)

    def on_train_batch_begin(self, batch, **kwargs):
        """Calls the `on_train_batch_begin` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
        """
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, **kwargs)

    def on_train_batch_end(self, batch, **kwargs):
        """Calls the `on_train_batch_end` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
        """
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, **kwargs)

    def on_validation_batch_begin(self, batch, **kwargs):
        """Calls the `on_validation_batch_begin` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
        """
        for callback in self.callbacks:
            callback.on_validation_batch_begin(batch, **kwargs)

    def on_validation_batch_end(self, batch, **kwargs):
        """Calls the `on_validation_batch_end` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
        """
        for callback in self.callbacks:
            callback.on_validation_batch_end(batch, **kwargs)

    def on_train_begin(self, **kwargs):
        """Calls the `on_train_begin` methods of its callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        """Calls the `on_train_end` methods of its callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_validation_begin(self, **kwargs):
        """Calls the `on_validation_begin` methods of its callbacks."""
        for callback in self.callbacks:
            callback.on_validation_begin(**kwargs)

    def on_validation_end(self, **kwargs):
        """Calls the `on_validation_end` methods of its callbacks."""
        for callback in self.callbacks:
            callback.on_validation_end(**kwargs)

    def __iter__(self):
        return iter(self.callbacks)
