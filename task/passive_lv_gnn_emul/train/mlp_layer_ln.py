from pkg.train.layer.mlp_layer import MLPLayer
from torch import nn
from pkg.train.module.activation import get_activation


class MLPLayerLN(MLPLayer):
    def _init_graph(self) -> None:
        mlp_layers = self.mlp_layers

        for i in range(len(self.unit_sizes) - 1):
            cur_layer_name = f"{self.prefix_name}_{self.layer_name}_l{i + 1}"

            # add fc layer
            mlp_layers.add_module(cur_layer_name, LinearWithLecunInitialize(self.unit_sizes[i], self.unit_sizes[i + 1]))

            # add activation
            if self.activation and i != len(self.unit_sizes) - 2:
                mlp_layers.add_module(f"{cur_layer_name}_ac", get_activation(self.activation))

        # add batch/layer norm
        if self.layer_norm:
            mlp_layers.add_module(f"{self.prefix_name}_{self.layer_name}_ln", nn.LayerNorm(self.unit_sizes[-1]))

    def forward(self, x):
        return self.mlp_layers(x)


class LinearWithLecunInitialize(nn.Linear):
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.xavier_uniform_(self.weight, gain=1)

