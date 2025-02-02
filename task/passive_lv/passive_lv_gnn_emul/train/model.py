from typing import Dict, Sequence

import torch
import torch.nn as nn

from common.constant import TRAIN_NAME
from pkg.dnn_utils.method import segment_sum
from pkg.train.model.base_model import BaseModule
from pkg.train.trainer.base_trainer import BaseTrainer
from pkg.utils.logs import init_logger
from task.passive_lv.passive_lv_gnn_emul.train.datasets import LvDataset
from task.passive_lv.passive_lv_gnn_emul.train.message_passing_layer import MessagePassingModule
from task.passive_lv.passive_lv_gnn_emul.train.mlp_layer_ln import MLPLayerLN

logger = init_logger("PassiveLvGNNEmul")

torch.manual_seed(753)
torch.set_printoptions(precision=8)


class PassiveLvGNNEmulTrainer(BaseTrainer):
    """Trainer class for the Passive Left Ventricle Graph Neural Network Emulator.

    This class handles the training process for a graph neural network that emulates the passive mechanical
    behavior of the left ventricle. It manages data loading, model creation, loss computation, and validation.

    Attributes:
        dataset_class (class): The dataset class to use for loading LV mesh data
        senders (torch.Tensor): Source nodes indices for edges in the mesh
        receivers (torch.Tensor): Target nodes indices for edges in the mesh
        real_node_indices (torch.Tensor): Indices of real (non-padding) nodes
        n_total_nodes (int): Total number of nodes in the mesh
        displacement_mean (torch.Tensor): Mean displacement values for normalization
        displacement_std (torch.Tensor): Standard deviation of displacements for normalization

    note: this model has exactly followed the model arch by https://github.com/dodaltuin/passive-lv-gnn-emul
    we re-write the arch from jax to pytorch and assign model weight based on jax version as baseline
    """

    dataset_class = LvDataset

    def __init__(self) -> None:
        super().__init__()

        self.task_train["init_weight_file_path"] = (
            f"{self.task_base['task_path']}/train/{self.task_train['init_weight_file_path']}"
            if "init_weight_file_path" in self.task_train
            else None
        )

        # config relative to dataset
        dataset_config = self.dataset_class(self.task_data, TRAIN_NAME)

        self.senders = dataset_config.get_senders()
        self.receivers = dataset_config.get_receivers()
        self.real_node_indices = dataset_config.get_real_node_indices()
        self.n_total_nodes = dataset_config.get_n_total_nodes()
        self.displacement_mean = dataset_config.get_displacement_mean()
        self.displacement_std = dataset_config.get_displacement_std()

    def create_model(self) -> None:
        self.model = PassiveLvGNNEmulModel(
            self.task_train, self.senders, self.receivers, self.real_node_indices, self.n_total_nodes
        )

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        return self.loss(outputs, labels.squeeze(dim=0))

    def compute_validation_loss(self, predictions: torch.Tensor, labels: torch.Tensor):
        return self.compute_loss(predictions * self.displacement_std + self.displacement_mean, labels)

    def compute_metrics(self, metrics_func: callable, predictions: torch.Tensor, labels: torch.Tensor):
        return metrics_func(predictions * self.displacement_std + self.displacement_mean, labels.squeeze(dim=0))

    def validation_step_check(self, epoch: int, is_last_epoch: bool) -> bool:
        if epoch <= 20 or epoch % 5 == 0 or is_last_epoch:
            return True
        else:
            return False


class PassiveLvGNNEmulModel(BaseModule):
    """Graph Neural Network model for emulating passive left ventricle mechanics.

    This model implements an encoder-processor-decoder architecture using message passing
    neural networks to predict displacement fields in the left ventricle mesh.

    Attributes:
        node_input_mlp_layer (dict): Configuration for node encoder MLP
        edge_input_mlp_layer (dict): Configuration for edge encoder MLP
        theta_input_mlp_layer (dict): Configuration for parameter encoder MLP
        message_passing_layer_config (dict): Configuration for message passing layers
        decoder_layer_config (dict): Configuration for decoder MLPs
        receivers (torch.Tensor): Target nodes for each edge
        n_total_nodes (int): Total number of nodes in the mesh
        real_node_indices (torch.Tensor): Indices of real (non-padding) nodes
    """

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
        """Initialize the GNN model with the given configuration and mesh topology.

        Args:
            config (Dict): Model configuration dictionary
            senders (Sequence[int]): Source nodes for each edge
            receivers (torch.tensor): Target nodes for each edge
            real_node_indices (Sequence[int]): Indices of real nodes
            n_total_nodes (int): Total number of nodes in the mesh
        """
        super().__init__(config, *args, **kwargs)

        # mlp layer config
        self.node_input_mlp_layer = config["node_input_mlp_layer"]
        self.node_input_mlp_layer["init_weight_file_path"] = config["init_weight_file_path"]

        self.edge_input_mlp_layer = config["edge_input_mlp_layer"]
        self.edge_input_mlp_layer["init_weight_file_path"] = config["init_weight_file_path"]

        self.theta_input_mlp_layer = config["theta_input_mlp_layer"]
        self.theta_input_mlp_layer["init_weight_file_path"] = config["init_weight_file_path"]

        self.message_passing_layer_config = config["message_passing_layer"]
        self.message_passing_layer_config["init_weight_file_path"] = config["init_weight_file_path"]

        self.decoder_layer_config = config["decoder_layer"]
        self.decoder_layer_config["mlp_layer"]["init_weight_file_path"] = config["init_weight_file_path"]

        # message passing config
        self.message_passing_layer_config["senders"] = senders
        self.message_passing_layer_config["receivers"] = receivers
        self.message_passing_layer_config["n_total_nodes"] = n_total_nodes
        self.message_passing_layer_config["init_weight_file_path"] = config["init_weight_file_path"]

        # other config
        self.receivers = receivers
        self.n_total_nodes = n_total_nodes
        self.real_node_indices = real_node_indices

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
        """Initialize all neural network components of the model.

        Creates encoder MLPs for nodes, edges, and parameters, decoder MLPs for each output
        dimension, and the message passing layers.
        """
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
        input_node = x["nodes"].squeeze(dim=0)  # shape: (batch_size, node, 1) => (node, 1)
        input_edge = x["edges"].squeeze(dim=0)  # shape: (batch_size, edge, 3) => (edge, 3)
        input_theta = x["theta_vals"]  # shape: (batch_size, fea)
        input_z_global = x["shape_coeffs"]  # shape: (batch_size, fea)

        # ====== Encoder:
        # encode vertices and edges
        node = self.node_encode_mlp_layer(input_node)  # shape: (node, emb)
        edge = self.edge_encode_mlp_layer(input_edge)  # shape: (edge, emb)

        # perform K rounds of message passing
        node, edge = self.message_passing_layer(node, edge)  # shape: (node, emb), (edge, emb)

        # aggregate incoming messages to each node
        incoming_message = segment_sum(edge, self.receivers, self.n_total_nodes)  # shape: (node, emb)

        # final local learned representation is a concatenation of vector embedding and incoming messages
        z_local = torch.concat((node, incoming_message), dim=-1)  # shape: (node, emb)

        # only need local representation for real nodes
        z_local = z_local[self.real_node_indices,]  # shape: (real node, emb)

        # encode global parameters theta
        z_theta = self.theta_encode_mlp_layer(input_theta)  # shape: (batch_size, fea) => (batch_size, emb)

        # tile global values (z_theta and optionally z_global) to each individual real node
        if input_z_global is None:
            globals_array = torch.tile(z_theta, (z_local.shape[0], 1))  # shape: (real node, emb)
        else:
            # stack z_global with z_theta if z_global is inputted
            global_embedding = torch.hstack(
                (z_theta, input_z_global)
            )  # shape: (batch_size, emb) + (batch_size, fea) => (batch_size, emb)
            globals_array = torch.tile(global_embedding, (z_local.shape[0], 1))  # shape: (real node, emb)

        # final learned representation is (z_theta, z_local) or (z_theta, z_global, z_local)
        final_representation = torch.hstack((globals_array, z_local))  # shape: (real node, emb)

        # ====== Decoder:
        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [
            decode_mlp(final_representation) for decode_mlp in self.decoder_layer
        ]  # shape: (real node, pred), (real node, pred)

        # concatenate the predictions of each individual decoder mlp
        Upred = torch.hstack(individual_mlp_predictions)  # shape: (real node, pred)

        return Upred
