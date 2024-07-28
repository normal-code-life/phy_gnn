from typing import Optional

import torch
import torch.nn as nn

from common.constant import TRAIN_NAME
from pkg.train.layer.pooling_layer import *  # noqa
from pkg.train.model.base_model import BaseModule
from pkg.train.trainer.base_trainer import BaseTrainer, TrainerConfig
from pkg.utils.logging import init_logger
from task.graph_sage_v2.data.datasets import GraphSageTrainDataset
from task.graph_sage_v2.train.mlp_layer_ln import MLPLayerLN

logger = init_logger("GraphSage")

torch.manual_seed(753)
torch.set_printoptions(precision=8)


class GraphSAGETrainer(BaseTrainer):
    dataset_class = GraphSageTrainDataset

    def __init__(self) -> None:
        config = TrainerConfig()

        logger.info(f"{config.get_config()}")

        super().__init__(config)

        # config relative to dataset
        dataset_config = self.dataset_class(self.task_data, TRAIN_NAME)

        self.displacement_mean = dataset_config.get_displacement_mean()
        self.displacement_std = dataset_config.get_displacement_std()

    def create_model(self) -> BaseModule:
        return GraphSAGEModel(self.task_train)

    def compute_validation_loss(self, predictions: torch.Tensor, labels: torch.Tensor):
        return self.compute_loss(predictions * self.displacement_std + self.displacement_mean, labels)

    def compute_metrics(self, metrics_func: callable, predictions: torch.Tensor, labels: torch.Tensor):
        return metrics_func(predictions * self.displacement_std + self.displacement_mean, labels)


class GraphSAGEModel(BaseModule):
    """https://github.com/raunakkmr/GraphSAGE."""

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        # hyper-parameter config
        self.neighbour_layers = config["neighbour_layers"]
        self.default_padding_value = config.get("default_padding_value", -1)

        # mlp layer config
        self.node_input_mlp_layer_config = config["node_input_mlp_layer"]
        self.edge_input_mlp_layer_config = config["edge_input_mlp_layer"]
        self.theta_input_mlp_layer_config = config["theta_input_mlp_layer"]
        self.decoder_layer_config = config["decoder_layer"]

        # message config
        self.message_passing_layer_config = config["message_passing_layer"]
        self.layer_threshold = self.message_passing_layer_config["layer_threshold"]
        self.layer_selected_node_num = self.message_passing_layer_config["layer_selected_node_num"]

        self.node_update_fn = nn.ModuleList()
        self.edge_update_fn = nn.ModuleList()

        self._init_graph()

    def get_config(self) -> Dict:
        base_config = super().get_config()

        mlp_config = {
            "node_input_mlp_layer": self.node_input_mlp_layer,
            "edge_input_mlp_layer": self.edge_input_mlp_layer,
            "theta_input_mlp_layer": self.theta_input_mlp_layer_config,
            "message_config": self.message_layer_config,
            "decoder_layer_config": self.decoder_layer_config,
        }

        return {**base_config, **mlp_config}

    def _init_graph(self):
        # 2 encoder mlp
        self.node_encode_mlp_layer = MLPLayerLN(self.node_input_mlp_layer_config, prefix_name="node_encode")
        self.edge_encode_mlp_layer = MLPLayerLN(self.edge_input_mlp_layer_config, prefix_name="edge_encode")

        # aggregator pooling
        agg_method = self.message_passing_layer_config["agg_method"]
        self.message_agg_pooling = globals()[agg_method](self.message_passing_layer_config["agg_layer"])

        for i in range(len(self.layer_threshold)):
            self.node_update_fn.append(
                MLPLayerLN(self.message_passing_layer_config["node_mlp_layer"], prefix_name=f"message_edge_{i}")
            )
            self.edge_update_fn.append(
                MLPLayerLN(self.message_passing_layer_config["edge_mlp_layer"], prefix_name=f"message_node_{i}")
            )

        # theta mlp
        self.theta_encode_mlp_layer = MLPLayerLN(self.theta_input_mlp_layer_config, prefix_name="theta_encode")

        # decoder MLPs
        decoder_layer_config = self.decoder_layer_config
        self.decoder_layer = nn.ModuleList(
            [
                MLPLayerLN(decoder_layer_config, prefix_name=f"decode_{i}")
                for i in range(decoder_layer_config["output_dim"])
            ]
        )

    def _node_preprocess(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_node_fea: torch.Tensor = x["node_features"]  # shape: (batch_size, node_num, node_feature_dim)
        input_node_coord: torch.Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)

        return torch.concat([input_node_fea, input_node_coord], dim=-1)

    def _edge_coord_preprocess(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # parse input data
        # === read node feature/coord and corresponding neighbours
        input_node_coord: torch.Tensor = x["node_coord"]  # shape: (batch_size, node_num, node_coord_dim)
        input_edge_indices: torch.Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)

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

        # === mask part of indices if the seq length is variance
        indices_coord_expanded_mask: torch.Tensor = torch.eq(indices_coord_expanded, self.default_padding_value)

        indices_coord_expanded_w_mask: torch.Tensor = torch.where(
            indices_coord_expanded_mask, torch.zeros_like(indices_coord_expanded), indices_coord_expanded
        )

        # === gather coord
        node_seq_coord: torch.Tensor = torch.gather(node_coord_expanded, 1, indices_coord_expanded_w_mask)

        # combine node data + seq data => edge data
        # shape: (batch_size, node_num, seq, node_coord_dim) =>
        # (batch_size, node_num, seq, 2 * node_coord_dim)
        edge_vertex_coord: torch.Tensor = torch.concat([node_coord_expanded, node_seq_coord], dim=-1)
        edge_coord: torch.Tensor = node_coord_expanded - node_seq_coord

        return torch.concat([edge_coord, edge_vertex_coord], dim=-1)

    def _edge_fea_preprocess(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # parse input data
        input_node_fea: torch.Tensor = x["node_features"]  # shape: (batch_size, node_num, node_feature_dim)
        input_edge_indices: torch.Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)

        fea_dim: int = input_node_fea.shape[-1]  # feature for each of the node
        seq: int = input_edge_indices.shape[-1]  # neighbours seq for each of the center node

        # === expand node feature to match indices shape
        # shape: (batch_size, node_num, node_feature_dim/node_coord_dim) =>
        # (batch_size, node_num, 1, node_feature_dim) =>
        # (batch_size, node_num, seq, node_feature_dim)
        node_fea_expanded: torch.Tensor = input_node_fea.unsqueeze(2).expand(-1, -1, seq, -1)

        # parse seq data
        # === expand indices to match feature shape
        # shape: (batch_size, node_num, seq) =>
        # (batch_size, node_num, seq, 1) =>
        # (batch_size, node_num, seq, node_feature_dim)
        indices_fea_expanded: torch.Tensor = input_edge_indices.unsqueeze(-1).expand(-1, -1, -1, fea_dim)

        # === mask part of indices if the seq length is variance
        indices_fea_expanded_mask: torch.Tensor = torch.eq(indices_fea_expanded, self.default_padding_value)

        indices_fea_expanded_w_mask: torch.Tensor = torch.where(
            indices_fea_expanded_mask, torch.zeros_like(indices_fea_expanded), indices_fea_expanded
        )
        # === gather feature/coord
        node_seq_fea: torch.Tensor = torch.gather(node_fea_expanded, 1, indices_fea_expanded_w_mask)

        # combine node data + seq data => edge data
        # shape: (batch_size, node_num, seq, node_feature_dim) =>
        # (batch_size, node_num, seq, 2 * node_feature_dim)
        edge_fea: torch.Tensor = torch.concat([node_fea_expanded, node_seq_fea], dim=-1)

        return edge_fea

    def _edge_emb(self, node_emb: torch.Tensor, selected_seq_node: torch.Tensor) -> torch.Tensor:
        emb_dim: int = node_emb.shape[-1]  # feature for each of the node
        seq: int = selected_seq_node.shape[-1]  # neighbours seq for each of the center node

        # === expand node feature to match indices shape
        # shape: (batch_size, node_num, emb) => (batch_size, node_num, 1, emb) => (batch_size, node_num, seq, emb)
        node_emb_expanded: torch.Tensor = node_emb.unsqueeze(2).expand(-1, -1, seq, -1)

        # parse seq data
        # === expand indices to match feature shape
        # shape: (batch_size, node_num, seq) =>
        # (batch_size, node_num, seq, 1) =>
        # (batch_size, node_num, seq, emb_dim)
        selected_seq_node_expanded: torch.Tensor = selected_seq_node.unsqueeze(-1).expand(-1, -1, -1, emb_dim)

        # === gather feature/coord
        selected_seq_node_emb = torch.gather(node_emb_expanded, 1, selected_seq_node_expanded)

        return torch.concat([node_emb_expanded, selected_seq_node_emb], dim=-1)

    def _edge_coord(self, input_node_coord: torch.Tensor, selected_seq_node: torch.Tensor) -> torch.Tensor:
        coord_dim: int = input_node_coord.shape[-1]  # coord for each of the node
        seq: int = selected_seq_node.shape[-1]  # neighbours seq for each of the center node

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
        selected_seq_node_expanded: torch.Tensor = selected_seq_node.unsqueeze(-1).expand(-1, -1, -1, coord_dim)

        # === gather coord
        node_seq_coord: torch.Tensor = torch.gather(node_coord_expanded, 1, selected_seq_node_expanded)

        # combine node data + seq data => edge data
        # shape: (batch_size, node_num, seq, node_coord_dim) =>
        # (batch_size, node_num, seq, 2 * node_coord_dim)
        edge_vertex_coord: torch.Tensor = torch.concat([node_coord_expanded, node_seq_coord], dim=-1)
        edge_coord: torch.Tensor = node_coord_expanded - node_seq_coord

        return torch.concat([edge_coord, edge_vertex_coord], dim=-1)

    def random_select_nodes(self, indices: torch.Tensor, layer: int) -> torch.Tensor:
        threshold_lower_bound = self.layer_threshold[layer - 1] if layer > 0 else 0
        threshold_upper_bound = self.layer_threshold[layer]
        selected_node_num = self.layer_selected_node_num[layer]

        batch_size, rows, cols = indices.shape

        random_indices = torch.randint(
            threshold_lower_bound, threshold_upper_bound, (batch_size, rows, selected_node_num), dtype=torch.int32
        )

        batch_indices = torch.arange(batch_size)[:, None, None]

        row_indices = torch.arange(rows)[None, :, None]

        return indices[batch_indices, row_indices, random_indices]

    def message_passing_layer(
            self, input_edge_indices: torch.Tensor, input_node_coord: torch.Tensor, node_emb: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        final_node_emb: Optional[torch.Tensor] = None

        for i in range(len(self.layer_selected_node_num)):
            selected_seq_node = self.random_select_nodes(input_edge_indices, i)  # shape: (batch_size, node_num, seq)

            edge_emb = self._edge_emb(node_emb, selected_seq_node)  # shape: (batch_size, node_num, seq, emb)
            edge_coord = self._edge_coord(input_node_coord, selected_seq_node)  # shape: (batch_size, node_num, seq, emb)

            edge_seq_emb_up = torch.concat([edge_emb, edge_coord], dim=-1)

            edge_seq_emb_up = self.edge_update_fn[i](edge_seq_emb_up)  # shape: (batch_size, node_num, seq, edge_emb)

            edge_emb_pooling = self.message_agg_pooling(edge_seq_emb_up)  # shape: (batch_size, node_num, edge_emb)

            node_emb_up = torch.concat([node_emb, edge_emb_pooling], dim=-1)

            node_emb_up = self.node_update_fn[i](node_emb_up)  # shape: (batch_size, node_num, node_emb)

            final_node_emb = final_node_emb + node_emb_up if final_node_emb is not None else node_emb_up

        return final_node_emb  # shape: (batch_size, node_num, node_emb)

    def forward(self, x: Dict[str, torch.Tensor]):
        # ====== Input data (squeeze to align to previous project)
        input_edge_indices: torch.Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)
        input_node_coord: torch.Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)
        input_theta: torch.Tensor = x["theta_vals"]  # shape: (batch_size, graph_feature)
        input_z_global: torch.Tensor = x["shape_coeffs"]  # shape: (batch_size, graph_feature)

        input_node = self._node_preprocess(x)  # shape: (batch_size, node_num, node_feature_dim+coord_dim)
        node_emb = self.node_encode_mlp_layer(input_node)  # shape: (batch_size, node_num, node_emb)

        # ====== message passing layer: Encoder & Aggregate
        z_local = self.message_passing_layer(input_edge_indices, input_node_coord, node_emb)

        # encode global parameters theta
        z_theta = self.theta_encode_mlp_layer(input_theta)  # shape: (batch_size, theta_feature)

        # tile global values (z_theta and optionally z_global) to each individual real node
        global_fea = torch.concat((z_theta, input_z_global), dim=-1)  # shape: (batch_size, emb)
        global_fea = global_fea.unsqueeze(dim=-2)  # shape: (batch_size, 1, emb)
        global_fea_expanded = torch.tile(global_fea, (1, z_local.shape[1], 1))  # shape: (batch_size, node_num, emb)

        encoding_emb = torch.concat((global_fea_expanded, z_local), dim=-1)  # shape: (batch_size, node_num, emb)

        # ====== Decoder:
        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [
            decode_mlp(encoding_emb) for decode_mlp in self.decoder_layer
        ]  # shape: List[(batch_size, node_num, 1)]

        # concatenate the predictions of each individual decoder mlp
        output = torch.concat(individual_mlp_predictions, dim=-1)  # shape: (batch_size, node_num, 1)
        return output
