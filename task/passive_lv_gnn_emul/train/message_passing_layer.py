from typing import Dict, Sequence
from pkg.utils.logging import init_logger
from pkg.train.model.base_model import BaseModuleConfig, BaseModule
from pkg.train.module.mlp_layer import MLPConfig, MLPModule
import torch
from pkg.tf_utils.method import segment_sum

logger = init_logger("message_passing")


class MessagePassingConfig(BaseModuleConfig):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)

        self.layer_name = "message_passing_layer"

        self.mlp_layer_config = config["mlp_layer"]

        self.senders = config["senders"]
        self.receivers = config["receivers"]
        self.n_total_nodes = config["n_total_nodes"]
        self.K = config["K"]

    def get_config(self):
        base_config = super().get_config()

        message_passing_config = {
            "mlp_layer_config": self.mlp_layer_config,
            "senders": self.senders,
            "receivers": self.receivers,
            "n_total_nodes": self.n_total_nodes,
            "K": self.K,
        }

        return {**base_config, **message_passing_config}


class MessagePassingModule(BaseModule):
    def __init__(self, config: MessagePassingConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self._init_graph(config)

    def _init_graph(self, config: MessagePassingConfig):
        self.node_update_fn = []
        self.edge_update_fn = []

        for i in range(self.config.K):
            self.node_update_fn.append(self._init_mlp_graph(("message_passing_node_%d" % i)))
            self.edge_update_fn.append(self._init_mlp_graph("message_passing_edge_%d" % i))

    def _init_mlp_graph(self, prefix_name):
        config = MLPConfig(self.config.mlp_layer_config, prefix_name=prefix_name)

        return MLPModule(config)

    def aggregate_incoming_messages(self, message, receivers: Sequence[int], n_nodes: int):
        r"""Sum aggregates incoming messages to each node.

        Performs the sum over incoming messages $\sum_{j \in \mathcal{N}_i} m_{ij}^k$
        from the processor stage of Algorithm 2 of the manuscript, for all nodes $i$ similtaneously
        """
        return segment_sum(message, receivers, n_nodes)

    def message_passing_block(self, node, edge, i):
        receivers = self.config.receivers
        senders = self.config.senders
        n_total_nodes = self.config.n_total_nodes

        # calculate messages along each directed edge with an edge feature vector assigned
        edge_input = torch.concat((edge, node[receivers], node[senders]), dim=-1)
        messages = self.edge_update_fn[i](edge_input)

        # aggregate incoming messages m_{ij} from nodes i to j where i > j
        received_messages_ij = self.aggregate_incoming_messages(messages, receivers, n_total_nodes)

        # aggregate incoming messages m_{ij} from nodes i to j where i < j
        # m_{ij} = -m_{ji} where i < j (momentum conservation property of the message passing)
        received_messages_ji = self.aggregate_incoming_messages(-messages, senders, n_total_nodes)

        # concatenate node representation with incoming messages and then update node representation
        V = self.node_update_fn[i](torch.concat((node, received_messages_ij + received_messages_ji)))

        # return updated node and edge representations with residual connection
        return node + V, edge + messages

    def forward(self, node, edge):
        # node, edge
        for i in range(self.config.K):
            node, edge = self.message_passing_block(node, edge, i)

        return node, edge
