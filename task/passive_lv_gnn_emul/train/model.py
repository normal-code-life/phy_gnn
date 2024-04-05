import os
import sys
from typing import Dict, Sequence

import torch
import torch.nn as nn

from common.constant import TRAIN_NAME
from pkg.dnn_utils.method import segment_sum
from pkg.train.model.base_model import BaseModule
from pkg.train.trainer.base_trainer import BaseTrainer, TrainerConfig
from pkg.utils import io
from pkg.utils.logging import init_logger
from task.passive_lv_gnn_emul.train.datasets import LvDataset
from task.passive_lv_gnn_emul.train.message_passing_layer import \
    MessagePassingModule
from task.passive_lv_gnn_emul.train.mlp_layer_ln import MLPLayerLN

logger = init_logger("PassiveLvGNNEmul")

torch.manual_seed(753)
torch.set_printoptions(precision=8)


class PassiveLvGNNEmulTrainer(BaseTrainer):
    dataset_class = LvDataset

    def __init__(self, config_path: str) -> None:
        config = TrainerConfig(config_path)

        logger.info(f"{config.get_config()}")

        super().__init__(config)

        # config relative to dataset
        dataset_config = self.dataset_class(self.task_data, TRAIN_NAME)

        self.senders = dataset_config.get_senders()
        self.receivers = dataset_config.get_receivers()
        self.real_node_indices = dataset_config.get_real_node_indices()
        self.n_total_nodes = dataset_config.get_n_total_nodes()
        self.displacement_mean = dataset_config.get_displacement_mean()
        self.displacement_std = dataset_config.get_displacement_std()

    def create_model(self) -> BaseModule:
        return PassiveLvGNNEmulModel(
            self.task_train, self.senders, self.receivers, self.real_node_indices, self.n_total_nodes
        )

    def compute_loss(self, outputs, labels):
        return self.loss(outputs, labels.squeeze(dim=0))

    def compute_validation_loss(self, outputs, labels):
        return self.compute_loss(outputs * self.displacement_std + self.displacement_mean, labels)


class PassiveLvGNNEmulModel(BaseModule):
    def __init__(
        self,
        config: Dict,
        senders: Sequence[int],
        receivers: torch.tensor,
        real_node_indices: Sequence[int],
        n_total_nodes: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)

        # mlp layer config
        self.node_input_mlp_layer = config["node_input_mlp_layer"]
        self.edge_input_mlp_layer = config["edge_input_mlp_layer"]
        self.theta_input_mlp_layer = config["theta_input_mlp_layer"]
        self.message_passing_layer_config = config["message_passing_layer"]
        self.decoder_layer_config = config["decoder_layer"]

        # message passing config
        self.message_passing_layer_config["senders"] = senders
        self.message_passing_layer_config["receivers"] = receivers
        self.message_passing_layer_config["n_total_nodes"] = n_total_nodes

        # other config
        self.receivers = receivers
        self.n_total_nodes = n_total_nodes
        self.real_node_indices = real_node_indices

        logger.info(f'Message passing steps: {config["message_passing_steps"]}')
        logger.info(f'Num. shape coeffs: {config["n_shape_coeff"]}')

        self._init_graph()

    def get_config(self) -> Dict:
        base_config = super().get_config()

        mlp_config = {
            "node_input_mlp_layer": self.node_input_mlp_layer,
            "edge_input_mlp_layer": self.edge_input_mlp_layer,
            "theta_input_mlp_layer": self.theta_input_mlp_layer,
            "message_passing_layer_config": self.message_passing_layer_config,
            "decoder_layer_config": self.decoder_layer_config,
            "receivers": self.receivers,
            "n_total_nodes": self.n_total_nodes,
            "real_node_indices": self.real_node_indices,
        }

        return {**base_config, **mlp_config}

    def _init_graph(self):
        # 3 encoder mlp
        self.node_encode_mlp_layer = MLPLayerLN(self.node_input_mlp_layer, prefix_name="node_encode")
        self.edge_encode_mlp_layer = MLPLayerLN(self.edge_input_mlp_layer, prefix_name="edge_encode")

        # theta mlp
        self.theta_encode_mlp_layer = MLPLayerLN(self.theta_input_mlp_layer, prefix_name="theta_encode")

        # decoder MLPs
        decoder_layer_config = self.decoder_layer_config
        self.decoder_layer = nn.ModuleList(
            [
                MLPLayerLN(decoder_layer_config["mlp_layer"], prefix_name=f"decode_{i}")
                for i in range(decoder_layer_config["output_dim"])
            ]
        )

        # 2K processor mlp
        self.message_passing_layer = MessagePassingModule(self.message_passing_layer_config)

    def forward(self, x):
        # ====== Input data (squeeze to align to previous project)
        input_node = x["nodes"].squeeze(dim=0)  # shape: (1, 126, 1) => (126, 1)
        input_edge = x["edges"].squeeze(dim=0)  # shape: (1, 440, 3) => (440, 3)
        input_theta = x["theta_vals"]  # shape: (1, 2)
        input_z_global = x["shape_coeffs"]  # shape: (1, 2)

        # ====== Encoder:
        # encode vertices and edges
        node = self.node_encode_mlp_layer(input_node)  # shape: (126, 40)
        edge = self.edge_encode_mlp_layer(input_edge)  # shape: (440, 40)

        # perform K rounds of message passing
        node, edge = self.message_passing_layer(node, edge)  # shape: (126, 40), (440, 40)

        # aggregate incoming messages to each node
        incoming_message = segment_sum(edge, self.receivers, self.n_total_nodes)  # shape: (126, 40)

        # final local learned representation is a concatenation of vector embedding and incoming messages
        z_local = torch.concat((node, incoming_message), dim=-1)  # shape: (126, 80)

        # only need local representation for real nodes
        z_local = z_local[self.real_node_indices,]  # shape: (96, 80)

        # encode global parameters theta
        z_theta = self.theta_encode_mlp_layer(input_theta)  # shape: (1, 2) => (1, 40)

        # tile global values (z_theta and optionally z_global) to each individual real node
        if input_z_global is None:
            globals_array = torch.tile(z_theta, (z_local.shape[0], 1))  # shape: (96, 40)
        else:
            # stack z_global with z_theta if z_global is inputted
            global_embedding = torch.hstack((z_theta, input_z_global))  # shape: (1, 40) + (1, 2) => (1, 42)
            globals_array = torch.tile(global_embedding, (z_local.shape[0], 1))  # shape: (96, 42)

        # final learned representation is (z_theta, z_local) or (z_theta, z_global, z_local)
        final_representation = torch.hstack((globals_array, z_local))  # shape: (96, 122)

        # ====== Decoder:
        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [
            decode_mlp(final_representation) for decode_mlp in self.decoder_layer
        ]  # shape: (96, 1), (96, 1)

        # concatenate the predictions of each individual decoder mlp
        Upred = torch.hstack(individual_mlp_predictions)  # shape: (96, 2)

        return Upred


if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    task_dir = io.get_cur_abs_dir(cur_path)
    model = PassiveLvGNNEmulTrainer(f"{task_dir}/train_config.yaml")
    model.train()
