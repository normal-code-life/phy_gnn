from typing import Tuple, Dict, List, Optional
import torch
from torch import Tensor
import numpy as np


class TFRecordToTensor:
    convert_type = {
        "float": torch.float32,
        "int": torch.int64,
    }

    def __init__(self, context_list: Dict[str, str], feature_list: Dict[str, str]) -> None:
        self.context_list = context_list
        self.feature_list = feature_list

    def __call__(self, sample: Tuple[Dict[str, List], Dict[str, List]]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Convert TFRecord to Tensor.
        """

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MaxMinNormalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"