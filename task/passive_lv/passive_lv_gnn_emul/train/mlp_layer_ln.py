from torch import nn

from pkg.train.layer.mlp_layer import MLPLayerBase
from pkg.train.module.activation import get_activation


class MLPLayerLN(MLPLayerBase):
    def _init_graph(self) -> None:
        """Initializes the MLP graph structure with fully connected layers, activations and layer normalization.

        This method builds the MLP by:
        1. Creating fully connected layers between each pair of consecutive unit sizes
        2. Adding activation functions after each layer except the last one
        3. Adding layer normalization at the end if layer_norm is True

        The layer names follow the pattern: {layer_name}_l{layer_number} for FC layers
        and {layer_name}_ln for the layer norm.
        """
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
