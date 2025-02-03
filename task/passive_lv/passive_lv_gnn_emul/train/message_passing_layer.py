from typing import Dict

import torch
from torch import nn

from pkg.dnn_utils.method import segment_sum
from pkg.train.model.base_model import BaseModule
from pkg.utils.logs import init_logger
from task.passive_lv.passive_lv_gnn_emul.train.mlp_layer_ln import MLPLayerLN

logger = init_logger("MessagePassing")


class MessagePassingModule(BaseModule):
    """Message passing neural network module for processing graph-structured data.

    This class implements a message passing neural network that processes node and edge features
    through multiple rounds of message passing. It is designed specifically for the Left Ventricle
    mesh data processing.

    Attributes:
        layer_name (str): Name identifier for the layer
        edge_mlp_layer_config (Dict): Configuration for edge MLP layers
        node_mlp_layer_config (Dict): Configuration for node MLP layers
        senders (torch.Tensor): Source nodes indices for edges
        receivers (torch.Tensor): Target nodes indices for edges
        n_total_nodes (int): Total number of nodes in the graph
        K (int): Number of message passing iterations
        node_update_fn (nn.ModuleList): List of node update MLPs
        edge_update_fn (nn.ModuleList): List of edge update MLPs
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)

        # === Config
        self.layer_name = "message_passing_layer"

        self.edge_mlp_layer_config = config["edge_mlp_layer"]
        self.edge_mlp_layer_config["init_weight_file_path"] = config["init_weight_file_path"]
        self.node_mlp_layer_config = config["node_mlp_layer"]
        self.node_mlp_layer_config["init_weight_file_path"] = config["init_weight_file_path"]

        self.senders = config["senders"]
        self.receivers = config["receivers"]
        self.n_total_nodes = config["n_total_nodes"]
        self.K = config["K"]

        # === Layers
        self.node_update_fn = nn.ModuleList()
        self.edge_update_fn = nn.ModuleList()

        self._init_graph()

    def get_config(self) -> Dict:
        base_config = super().get_config()

        message_passing_config = {
            "edge_mlp_layer_config": self.edge_mlp_layer_config,
            "node_mlp_layer_config": self.node_mlp_layer_config,
            "senders": self.senders,
            "receivers": self.receivers,
            "n_total_nodes": self.n_total_nodes,
            "K": self.K,
        }

        return {**base_config, **message_passing_config}

    def _init_graph(self) -> None:
        for i in range(self.K):
            self.node_update_fn.append(MLPLayerLN(self.node_mlp_layer_config, prefix_name=f"message_passing_node_{i}"))
            self.edge_update_fn.append(MLPLayerLN(self.edge_mlp_layer_config, prefix_name=f"message_passing_edge_{i}"))

    @staticmethod
    def aggregate_incoming_messages(message, receivers: torch.tensor, n_nodes: int):
        r"""Sum aggregates incoming messages to each node.

        Performs the sum over incoming messages $\sum_{j \in \mathcal{N}_i} m_{ij}^k$
        from the processor stage of Algorithm 2 of the manuscript, for all nodes $i$ similtaneously
        """
        return segment_sum(message, receivers, n_nodes)

    def message_passing_block(self, node, edge, i):
        receivers = self.receivers
        senders = self.senders
        n_total_nodes = self.n_total_nodes

        # calculate messages along each directed edge with an edge feature vector assigned
        edge_input_receivers = torch.concat((edge, node[senders], node[receivers]), dim=-1)  # shape: (edge, emb)
        edge_input_senders = torch.concat((-edge, node[receivers], node[senders]), dim=-1)  # shape: (edge, emb)
        messages_receivers = self.edge_update_fn[i](edge_input_receivers)  # shape: (edge, emb)
        messages_senders = self.edge_update_fn[i](edge_input_senders)  # shape: (edge, emb)

        received_messages_ij = self.aggregate_incoming_messages(
            messages_receivers, receivers, n_total_nodes
        )  # shape: (node, emb)
        received_messages_ji = self.aggregate_incoming_messages(
            messages_senders, senders, n_total_nodes
        )  # shape: (node, emb)

        # concatenate node representation with incoming messages and then update node representation
        node_input = torch.concat((node, received_messages_ij + received_messages_ji), dim=-1)  # shape: (node, emb)
        V = self.node_update_fn[i](node_input)  # shape: (node, emb)

        # return updated node and edge representations with residual connection
        return node + V, edge + messages_receivers + messages_senders

    def forward(self, node, edge):
        # node, edge
        for i in range(self.K):
            node, edge = self.message_passing_block(node, edge, i)

        return node, edge
