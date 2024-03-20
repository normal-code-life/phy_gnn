from pkg.train.layer.mlp_layer import MLPConfig, MLPLayer
from torch import nn
from pkg.train.module.activation import get_activation


class MLPLayerLN(MLPLayer):
    def _init_graph(self, config: MLPConfig):
        sequential = nn.Sequential()

        for i in range(len(config.unit_sizes) - 1):
            cur_layer_name = f"{config.prefix_name}_{config.layer_name}_l{i + 1}"

            # add fc layer
            sequential.add_module(cur_layer_name, nn.Linear(config.unit_sizes[i], config.unit_sizes[i + 1]))

            # add activation
            if config.activation and i != len(config.unit_sizes) - 2:
                sequential.add_module(f"{cur_layer_name}_ac", get_activation(config.activation))

        # add batch/layer norm
        if config.layer_norm:
            sequential.add_module(f"{config.prefix_name}_{config.layer_name}_ln", nn.LayerNorm(config.unit_sizes[-1]))

        return sequential

    def forward(self, x):
        return self.mlp_layers(x)
