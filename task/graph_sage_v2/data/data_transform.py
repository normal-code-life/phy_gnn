from pkg.train.module.data_transform import DataTransform
from typing import Tuple, Dict

from torch import Tensor


class ConvertDataDim(DataTransform):
    def __init__(self, config: Dict) -> None:
        self.feature_config = config

    def __call__(self, sample: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        context, feature = sample

        for name, fea in context.items():
            if name in self.feature_config:
                context[name] = fea.squeeze(dim=self.feature_config[name])

        for name, fea in feature.items():
            if name in self.feature_config:
                feature[name] = fea.squeeze(dim=self.feature_config[name])

        return context, feature
