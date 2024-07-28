import abc
from typing import Tuple, Dict, List, Optional
import torch
from torch import Tensor
import numpy as np


max_val_name = "max_val"
mim_val_name = "min_val"


class DataTransform(abc.ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"please implement this function __call__(self, *args, **kwargs)")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TFRecordToTensor(DataTransform):
    convert_type = {
        "float": torch.float32,
        "int": torch.int64,
    }

    def __init__(self, config: Dict) -> None:
        self.context_list = config["context_description"]
        self.feature_list = config["feature_description"]

    def __call__(self, sample: Tuple[Dict[str, List], Dict[str, List]]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        context, feature = sample

        context_tensor: Dict[str, Tensor] = dict()
        feature_tensor: Dict[str, Tensor] = dict()

        for name, fea in context.items():
            if name not in self.context_list:
                raise ValueError(f"please check your feature list and add {name}")

            context_tensor[name] = torch.tensor(
                np.array(fea), dtype=self.convert_type[self.context_list[name]]
            )

        for name, fea in feature.items():
            if name not in self.feature_list:
                raise ValueError(f"please check your feature list and add {name}")

            feature_tensor[name] = torch.tensor(
                np.array(fea), dtype=self.convert_type[self.feature_list[name]]
            )

        return context_tensor, feature_tensor


class TensorToGPU(DataTransform):
    def __init__(self, config: Dict) -> None:
        self.gpu = config["gpu"]
        self.cuda_core = config["cuda_core"]

    def __call__(self, sample: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        if not self.gpu:
            return sample

        context, feature = sample

        for name, fea in context.items():
            context[name] = fea.cuda(device=self.cuda_core)

        for name, fea in feature.items():
            feature[name] = fea.cuda(device=self.cuda_core)

        return context, feature


class MaxMinNormalize(DataTransform):
    """Normalize a tensor with max and min.
    """

    def __init__(self, config: Dict) -> None:
        self.feature_config = config

    def __call__(self, sample: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        context, feature = sample

        for name, fea in context.items():
            if name in self.feature_config:
                max_val = self.feature_config[name][max_val_name]
                min_val = self.feature_config[name][mim_val_name]
                context[name] = self._normal_max_min_transform(fea, max_val, min_val)

        for name, fea in feature.items():
            if name in self.feature_config:
                max_val = self.feature_config[name][max_val_name]
                min_val = self.feature_config[name][mim_val_name]
                feature[name] = self._normal_max_min_transform(fea, max_val, min_val)

        return context, feature

    @staticmethod
    def _normal_max_min_transform(
            array: Tensor, max_norm_val: Tensor, min_norm_val: Tensor
    ) -> Tensor:
        return (array - min_norm_val) / (max_norm_val - min_norm_val)


class CovertToModelInputs(DataTransform):
    def __init__(self, config: Dict) -> None:
        self.labels_name = config["labels"]

    def __call__(self, sample: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], Tensor]:
        context, feature = sample

        inputs: Optional[Dict[str, Tensor]] = dict()
        labels: Optional[Tensor] = None

        for name, fea in context.items():
            if name not in self.labels_name:
                inputs[name] = fea
            else:
                labels = fea

        for name, fea in feature.items():
            if name not in self.labels_name:
                inputs[name] = fea
            else:
                labels = fea

        return inputs, labels
