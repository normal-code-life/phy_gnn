from typing import Dict

import torch
import torch.nn as nn

from pkg.train.layer.pooling_layer import MeanAggregator, SUMAggregator  # noqa
from pkg.train.model.base_model import BaseModule
from pkg.train.trainer.base_trainer import BaseTrainer, TrainerConfig
from pkg.utils.logs import init_logger
from task.passive_biv.data.datasets_train_hdf5 import FEHeartSageTrainDataset
from task.passive_biv.utils.module.mlp_layer_ln import MLPLayer

logger = init_logger("FEPassiveBivHeartSage")

torch.manual_seed(753)
torch.set_printoptions(precision=8)


class TestTrainer(BaseTrainer):
    dataset_class = FEHeartSageTrainDataset

    def __init__(self) -> None:
        config = TrainerConfig()

        logger.info(f"{config.get_config()}")

        super().__init__(config)

    def create_model(self) -> None:
        self.model = FEHeartSAGEModel(self.task_train)

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = dict()

        for label_name in self.labels:
            prediction = predictions[label_name]
            label = labels[label_name]

            _, node_num, _ = prediction.shape
            _, label_node_num, _ = label.shape

            if node_num != label_node_num:
                selected_node = labels["selected_node"][0, :]
                label = torch.index_select(label, 1, selected_node)

            losses[label_name] = self.loss(prediction, label)

        return losses


class FEHeartSAGEModel(BaseModule):
    """https://github.com/raunakkmr/GraphSAGE."""

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        # hyper-parameter config
        self.select_node_num = config["select_node_num"]
        self.select_edge_num = config["select_edge_num"]
        self.default_padding_value = config.get("default_padding_value", -1)

        # mlp layer config
        self.node_input_mlp_layer_config = config["node_input_mlp_layer"]
        self.theta_input_mlp_layer_config = config["theta_input_mlp_layer"]
        self.decoder_layer_config = config["decoder_layer"]

        # message config
        self.message_passing_layer_config = config["message_passing_layer"]
        self.message_layer_num = self.message_passing_layer_config["message_layer_num"]

        self.node_update_fn = nn.ModuleList()
        self.edge_update_fn = nn.ModuleList()

        self._init_graph()

    def get_config(self) -> Dict:
        base_config = super().get_config()

        mlp_config = {
            "node_input_mlp_layer": self.node_input_mlp_layer,
            "theta_input_mlp_layer": self.theta_input_mlp_layer_config,
            "message_config": self.message_layer_config,
            "decoder_layer_config": self.decoder_layer_config,
        }

        return {**base_config, **mlp_config}

    def _init_graph(self):
        # 2 encoder mlp
        self.node_encode_mlp_layer = MLPLayer(self.node_input_mlp_layer_config, prefix_name="node_encode")

        # aggregator pooling
        agg_method = self.message_passing_layer_config["agg_method"]
        self.message_agg_pooling = globals()[agg_method](self.message_passing_layer_config["agg_layer"])

        for i in range(self.message_layer_num):
            self.node_update_fn.append(
                MLPLayer(self.message_passing_layer_config["node_mlp_layer"], prefix_name=f"message_node_{i}")
            )
            self.edge_update_fn.append(
                MLPLayer(self.message_passing_layer_config["edge_mlp_layer"], prefix_name=f"message_edge_{i}")
            )

        # theta mlp
        self.theta_encode_mlp_layer = MLPLayer(self.theta_input_mlp_layer_config, prefix_name="theta_encode")

        # decoder MLPs
        decoder_layer_config = self.decoder_layer_config
        self.decoder_layer = nn.ModuleList(
            [
                MLPLayer(decoder_layer_config, prefix_name=f"decode_{i}")
                for i in range(decoder_layer_config["output_dim"])
            ]
        )

    def _node_preprocess(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # input_node_coord: torch.Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)
        input_laplace_coord: torch.Tensor = x["laplace_coord"]  # shape: (batch_size, node_num, coord_dim)
        input_node_fea: torch.Tensor = x["fiber_and_sheet"]  # shape: (batch_size, node_num, node_feature_dim)

        return torch.concat([input_laplace_coord, input_node_fea], dim=-1)

    def _edge_emb(self, node_emb: torch.Tensor, input_edge_indices: torch.Tensor) -> torch.Tensor:
        emb_dim: int = node_emb.shape[-1]  # feature for each of the node
        seq: int = input_edge_indices.shape[-1]  # neighbours seq for each of the center node

        # === expand node feature to match indices shape
        # shape: (batch_size, node_num, emb) =>
        # (batch_size, node_num, 1, emb) =>
        # (batch_size, node_num, seq, emb)
        node_emb_expanded: torch.Tensor = node_emb.unsqueeze(2).expand(-1, -1, seq, -1)

        # parse seq data
        # === expand indices to match feature shape
        # shape: (batch_size, node_num, seq) =>
        # (batch_size, node_num, seq, 1) =>
        # (batch_size, node_num, seq, emb)
        edge_seq_indices: torch.Tensor = input_edge_indices.unsqueeze(-1).expand(-1, -1, -1, emb_dim)

        # === gather feature/coord
        return torch.gather(node_emb_expanded, 1, edge_seq_indices)

    def _edge_coord(
        self, input_node_coord: torch.Tensor, input_edge_indices: torch.Tensor, selected_node: torch.Tensor
    ) -> torch.Tensor:
        coord_dim: int = input_node_coord.shape[-1]  # coord for each of the node
        seq: int = input_edge_indices.shape[-1]  # neighbours seq for each of the center node

        # === expand node feature to match indices shape
        # shape: (batch_size, node_num, node_coord_dim) =>
        # (batch_size, node_num, 1, node_coord_dim) =>
        # (batch_size, node_num, seq, node_coord_dim)
        node_coord_expanded: torch.Tensor = input_node_coord.unsqueeze(2).expand(-1, -1, seq, -1)

        # parse seq data
        # === expand indices to match feature shape
        # shape: (batch_size, node_num, seq) =>
        # (batch_size, node_num, seq, 1) =>
        # (batch_size, node_num, seq, node_coord_dim)
        indices_coord_expanded: torch.Tensor = input_edge_indices.unsqueeze(-1).expand(-1, -1, -1, coord_dim)

        # === gather coord
        node_seq_coord: torch.Tensor = torch.gather(node_coord_expanded, 1, indices_coord_expanded)

        # === select coord
        node_coord_expanded = node_coord_expanded[:, selected_node, :, :]

        # combine node data + seq data => edge data
        # shape: (batch_size, node_num, seq, node_coord_dim) =>
        # (batch_size, node_num, seq, 2 * node_coord_dim)
        edge_vertex_coord: torch.Tensor = torch.concat([node_coord_expanded, node_seq_coord], dim=-1)
        edge_coord: torch.Tensor = node_coord_expanded - node_seq_coord

        return torch.concat([edge_coord, edge_vertex_coord], dim=-1)

    def random_select_nodes(self, indices: torch.Tensor, select_node: torch.Tensor) -> torch.Tensor:
        batch_size, node_num, seq_num = indices.shape

        select_batch = torch.arange(batch_size)

        select_node_num = self.select_node_num if self.training else node_num

        select_indices = torch.randint(
            0, seq_num, (batch_size, select_node_num, self.select_edge_num), dtype=torch.int64
        )

        selected_edge = indices[
            select_batch[:, None, None], select_node[None, :, None], select_indices
        ]  # shape: (batch_size, node_num, seq)

        return selected_edge

    def message_passing_layer(self, x: Dict, node_emb: torch.Tensor) -> torch.Tensor:
        input_edge_indices: torch.Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)
        input_node_coord: torch.Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)
        selected_node: torch.Tensor = x["selected_node"][0, :]  # shape: (batch_size, node_num)

        _, node_num, _ = input_edge_indices.shape
        selected_node = selected_node if self.training else torch.arange(node_num)

        selected_edge = self.random_select_nodes(input_edge_indices, selected_node)

        # shape: (batch_size, node_num, 1, node_emb) => (batch_size, node_num, seq, node_emb)
        node_self_emb = node_emb[:, selected_node, :]
        node_self_seq_emb = node_self_emb.unsqueeze(dim=-2).expand(-1, -1, self.select_edge_num, -1)

        # shape: (batch_size, node_num, seq, node_emb)
        node_edge_seq_emb = self._edge_emb(node_emb, selected_edge)

        # shape: (batch_size, node_num, seq, coord) -> (batch_size, node_num, seq, edge_emb)
        edge_coord = self._edge_coord(input_node_coord, selected_edge, selected_node)
        edge_coord_seq_emb = self.edge_update_fn[0](edge_coord)

        emb_concat = torch.concat([node_self_seq_emb, node_edge_seq_emb, edge_coord_seq_emb], dim=-1)

        node_emb_up = self.node_update_fn[0](emb_concat)  # shape: (batch_size, node_num, seq, node_emb)

        node_emb_pooling = self.message_agg_pooling(node_emb_up)  # shape: (batch_size, node_num, node_emb)

        return node_self_emb + node_emb_pooling

    def forward(self, x: Dict[str, torch.Tensor]):
        # ====== Input data (squeeze to align to previous project)
        mat_param: torch.Tensor = x["mat_param"]  # shape: (batch_size, mat_param)
        pressure: torch.Tensor = x["pressure"]  # shape: (batch_size, pressure)
        input_shape_coeffs: torch.Tensor = x["shape_coeffs"]  # shape: (batch_size, graph_feature)

        input_node = self._node_preprocess(x)  # shape: (batch_size, node_num, node_feature_dim+coord_dim)

        # ====== message passing layer: Encoder & Aggregate
        # node_emb = self.node_encode_mlp_layer(input_node)  # shape: (batch_size, node_num, node_emb)

        # z_local = self.message_passing_layer(x, node_emb)  # shape: (batch_size, node_num, node_emb)

        # ====== Decoder:
        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [
            decode_mlp(input_node) for decode_mlp in self.decoder_layer
        ]  # shape: List[(batch_size, node_num, 1)]

        # concatenate the predictions of each individual decoder mlp
        output = dict()

        output["displacement"] = torch.concat(individual_mlp_predictions, dim=-1)  # shape: (batch_size, node_num, 1)

        return output
