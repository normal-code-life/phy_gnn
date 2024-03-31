from pkg.train.layer.mlp_layer import MLPLayer
from torch import nn
from pkg.train.module.activation import get_activation
from pkg.utils.logging import init_logger


class MLPLayerLN(MLPLayer):
    def _init_graph(self) -> None:
        for i in range(len(self.unit_sizes) - 1):
            cur_layer_name = f"{self.prefix_name}_{self.layer_name}_l{i + 1}"

            # add fc layer
            self._init_fc(cur_layer_name, i)

            # add activation
            if self.activation and i != len(self.unit_sizes) - 2:
                self.mlp_layers.add_module(f"{cur_layer_name}_ac", get_activation(self.activation))

        # add batch/layer norm
        if self.layer_norm:
            self.mlp_layers.add_module(
                f"{self.prefix_name}_{self.layer_name}_ln", nn.LayerNorm(self.unit_sizes[-1], eps=1e-6)
            )

    def forward(self, x):
        return self.mlp_layers(x)
