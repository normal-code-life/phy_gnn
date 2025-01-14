from torch import nn

from pkg.train.layer.mlp_layer import MLPLayerBase
from pkg.train.module.activation import get_activation


class MLPLayerLN(MLPLayerBase):
    def _init_graph(self) -> None:
        for i in range(len(self.unit_sizes) - 1):
            cur_layer_name = f"{self.layer_name}_l{i + 1}"

            # add fc layer
            self._init_fc(cur_layer_name, self.unit_sizes[i], self.unit_sizes[i + 1])

            # add activation
            if self.activation and i != len(self.unit_sizes) - 2:
                self.mlp_layers.add_module(f"{cur_layer_name}_ac", get_activation(self.activation))

        # add batch/layer norm
        if self.layer_norm:
            self.mlp_layers.add_module(f"{self.layer_name}_ln", nn.LayerNorm(self.unit_sizes[-1], eps=1e-6))
