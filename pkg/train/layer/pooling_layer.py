import math
import numpy as np
import torch
import torch.nn as nn
from pkg.train.model.base_model import BaseModule
from typing import Dict


# class Aggregator(nn.Module):
#
#     def __init__(self, input_dim=None, output_dim=None, device='cpu'):
#         """
#         Parameters
#         ----------
#         input_dim : int or None.
#             Dimension of input node features. Used for defining fully
#             connected layer in pooling aggregators. Default: None.
#         output_dim : int or None
#             Dimension of output node features. Used for defining fully
#             connected layer in pooling aggregators. Currently only works when
#             input_dim = output_dim. Default: None.
#         """
#         super(Aggregator, self).__init__()
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.device = device
#
#     def forward(self, features, nodes, mapping, rows, num_samples=25):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             An (n' x input_dim) tensor of input node features.
#         nodes : numpy array
#             nodes is a numpy array of nodes in the current layer of the computation graph.
#         mapping : dict
#             mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
#             its position in the layer of nodes in the computationn graph
#             before nodes. For example, if the layer before nodes is [2,5],
#             then mapping[2] = 0 and mapping[5] = 1.
#         rows : numpy array
#             rows[i] is an array of neighbors of node i which is present in nodes.
#         num_samples : int
#             Number of neighbors to sample while aggregating. Default: 25.
#
#         Returns
#         -------
#         out : torch.Tensor
#             An (len(nodes) x output_dim) tensor of output node features.
#             Currently only works when output_dim = input_dim.
#         """
#         _choice, _len, _min = np.random.choice, len, min
#         mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
#         if num_samples == -1:
#             sampled_rows = mapped_rows
#         else:
#             sampled_rows = [_choice(row, _min(_len(row), num_samples), _len(row) < num_samples) for row in mapped_rows]
#
#         n = _len(nodes)
#         if self.__class__.__name__ == 'LSTMAggregator':
#             out = torch.zeros(n, 2*self.output_dim).to(self.device)
#         else:
#             out = torch.zeros(n, self.output_dim).to(self.device)
#         for i in range(n):
#             if _len(sampled_rows[i]) != 0:
#                 out[i, :] = self._aggregate(features[sampled_rows[i], :])
#
#         return out
#
#     def _aggregate(self, features):
#         """
#         Parameters
#         ----------
#
#         Returns
#         -------
#         """
#         raise NotImplementedError
#
#
# # class MeanAggregator(Aggregator):
# #
# #     def _aggregate(self, features):
# #         """
# #         Parameters
# #         ----------
# #         features : torch.Tensor
# #             Input features.
# #
# #         Returns
# #         -------
# #         Aggregated feature.
# #         """
# #         return torch.mean(features, dim=0)
#
#
# class PoolAggregator(Aggregator):
#
#     def __init__(self, input_dim, output_dim, device='cpu'):
#         """
#         Parameters
#         ----------
#         input_dim : int
#             Dimension of input node features. Used for defining fully connected layer.
#         output_dim : int
#             Dimension of output node features. Used for defining fully connected layer. Currently only works when output_dim = input_dim.
#         """
#         super(PoolAggregator, self).__init__(input_dim, output_dim, device)
#
#         self.fc1 = nn.Linear(input_dim, output_dim)
#         self.relu = nn.ReLU()
#
#     def _aggregate(self, features):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             Input features.
#
#         Returns
#         -------
#         Aggregated feature.
#         """
#         out = self.relu(self.fc1(features))
#         return self._pool_fn(out)
#
#     def _pool_fn(self, features):
#         """
#         Parameters
#         ----------
#
#         Returns
#         -------
#         """
#         raise NotImplementedError
#
#
# class MaxPoolAggregator(PoolAggregator):
#
#     def _pool_fn(self, features):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             Input features.
#
#         Returns
#         -------
#         Aggregated feature.
#         """
#         return torch.max(features, dim=0)[0]
#
#
# class MeanPoolAggregator(PoolAggregator):
#
#     def _pool_fn(self, features):
#         """
#         Parameters
#         ----------
#         features : torch.Tensor
#             Input features.
#
#         Returns
#         -------
#         Aggregated feature.
#         """
#         return torch.mean(features, dim=0)


class PoolingLayer(BaseModule):
    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super(PoolingLayer, self).__init__(config, *args, **kwargs)

        self.agg_dim = config.get("agg_dim", -1)
        self.keep_dim = config.get("keep_dim", True)

        # self.pool_fn = pool_fn
        # self.seq_len = seq_len
        # self.emb_size = emb_size
        #
        # self.base_weight = tf.convert_to_tensor([a * math.exp(-b * t) + c for t in range(self.seq_len)])  # (seq_len,)

    # def call(self, inputs, mask=None, training=None, decay=None):
    #     #  inputs: (batch, seq_len, output_size)
    #     #  mask: (batch, seq_len)
    #     #  decay: (seq_len,)
    #
    #     if self.pool_fn == PoolingFn.MAX.value:
    #         mask = tf.repeat(tf.expand_dims(mask, axis=-1), self.emb_size, axis=-1)  # (batch, seq_len, output_size)
    #         outputs = tf.multiply(inputs, mask)  # (batch, seq_len, output_size)
    #         outputs = tf.reduce_max(outputs, axis=1, keepdims=False)  # (batch, output_size)
    #
    #     elif self.pool_fn == PoolingFn.SUM.value:
    #         weight = tf.expand_dims(mask, axis=-1)  # (batch, seq_len, 1)
    #         weight = tf.repeat(weight, self.emb_size, axis=-1)  # (batch, seq_len, output_size)
    #         outputs = tf.reduce_sum(tf.multiply(inputs, weight), axis=1, keepdims=False)  # (batch, output_size)
    #
    #     elif self.pool_fn == PoolingFn.AVG.value:
    #         weight = tf.math.divide(mask, tf.reduce_sum(tf.add(mask, EPS), axis=1, keepdims=True))  # (batch, seq_len)
    #         weight = tf.expand_dims(weight, axis=-1)  # (batch, seq_len, 1)
    #         weight = tf.repeat(weight, self.emb_size, axis=-1)  # (batch, seq_len, output_size)
    #         outputs = tf.reduce_sum(tf.multiply(inputs, weight), axis=1, keepdims=False)  # (batch, output_size)
    #
    #     elif self.pool_fn == PoolingFn.POS_DECAY_SUM.value:
    #         weight = tf.tile(tf.expand_dims(self.base_weight, 0), multiples=[tf.shape(mask)[0], 1])  # (batch, seq_len)
    #         weight = tf.multiply(weight, tf.add(mask, EPS))  # (batch, seq_len)
    #         weight = tf.expand_dims(weight, -1)  # (batch, seq_len, 1)
    #         weight = tf.tile(weight, multiples=[1, 1, self.emb_size])  # (batch, seq_len, emb)
    #         outputs = tf.reduce_sum(tf.multiply(inputs, weight), axis=1, keepdims=False)  # (batch, output_size)
    #
    #     elif self.pool_fn == PoolingFn.POS_DECAY_AVG.value:
    #         weight = tf.tile(tf.expand_dims(self.base_weight, 0), multiples=[tf.shape(mask)[0], 1])  # (batch, seq_len)
    #         weight = tf.multiply(weight, tf.add(mask, EPS))  # (batch, seq_len)
    #         weight = tf.math.divide(weight, tf.reduce_sum(weight, axis=1, keepdims=True))  # (batch, seq_len)
    #         weight = tf.expand_dims(weight, -1)  # (batch, seq_len, 1)
    #         weight = tf.tile(weight, multiples=[1, 1, self.emb_size])  # (batch, seq_len, emb)
    #         outputs = tf.reduce_sum(tf.multiply(inputs, weight), axis=1, keepdims=False)  # (batch, output_size)
    #
    #     else:
    #         raise ValueError("please define the right pooling function")
    #
    #     return outputs


class MeanAggregator(PoolingLayer):
    def __init__(self, config: Dict, *args, **kwargs) -> None:
        config["prefix_name"] = "mean_pooling_agg"

        super(MeanAggregator, self).__init__(config, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        return torch.mean(x, dim=0)


class SUMAggregator(PoolingLayer):
    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super(SUMAggregator, self).__init__(config, *args, **kwargs)
        self.prefix_name = "sum_pooling_agg"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        return torch.sum(x, dim=self.agg_dim, keepdim=self.keep_dim)

