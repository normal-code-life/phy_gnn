from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor

from pkg.train.layer.pooling_layer import MeanAggregator, SUMAggregator  # noqa
from pkg.train.model.base_model import BaseModule
from pkg.train.trainer.base_trainer import BaseTrainer, TrainerConfig
from pkg.utils.logs import init_logger
from task.passive_biv.data.datasets_train_hdf5 import FEHeartSageTrainDataset
from task.passive_biv.utils.module.mlp_layer_ln import MLPLayerV2

logger = init_logger("FEHeartSage")

torch.manual_seed(753)
torch.set_printoptions(precision=8)


class FEHeartSageV4Trainer(BaseTrainer):
    dataset_class = FEHeartSageTrainDataset

    def __init__(self) -> None:
        config = TrainerConfig()

        logger.info(f"{config.get_config()}")

        super().__init__(config)

        self.selected_node_num = self.task_train["select_node_num"]

        self.device = "cuda" if self.gpu else "cpu"

    def create_model(self) -> None:
        self.model = FEHeartSAGEModel(self.task_train)

    def validation_step_check(self, epoch: int, is_last_epoch: bool) -> bool:
        if epoch <= 20 or epoch % 5 == 0 or is_last_epoch:
            return True
        else:
            return False

    def post_transform_data(self, data: (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor])) -> (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor]):
        inputs, labels = super().post_transform_data(data)

        _, node_num, _ = inputs["edges_indices"].shape

        selected_node = torch.randint(
            0, node_num, size=(self.selected_node_num, ), dtype=torch.int64, device=self.device
        )

        selected_node_num = torch.tensor(self.selected_node_num, dtype=torch.int64, device=self.device)

        inputs["selected_node"] = selected_node
        labels["selected_node"] = selected_node

        inputs["selected_node_num"] = selected_node_num
        labels["selected_node_num"] = selected_node_num

        return inputs, labels

    def post_transform_val_data(self, data: (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor])) -> (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor]):
        inputs, labels = super().post_transform_val_data(data)

        _, node_num, _ = inputs["edges_indices"].shape

        selected_node = torch.arange(node_num, device=self.device)

        selected_node_num = node_num

        inputs["selected_node"] = selected_node
        labels["selected_node"] = selected_node

        inputs["selected_node_num"] = selected_node_num
        labels["selected_node_num"] = selected_node_num

        return inputs, labels

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
        self.input_layer_config = config["input_layer"]
        # self.node_input_mlp_layer_config = config["node_input_mlp_layer"]
        self.theta_input_mlp_layer_config = config["theta_input_mlp_layer"]
        self.decoder_layer_config = config["decoder_layer"]

        # message config
        self.message_passing_layer_config = config["message_passing_layer"]

        self.gpu = config.get("gpu", False)
        self.device = "cuda" if self.gpu else "cpu"

        self._init_graph()

    def get_config(self) -> Dict:
        base_config = super().get_config()

        mlp_config = {
            "node_input_mlp_layer": self.node_input_mlp_layer,
            "theta_input_mlp_layer": self.theta_input_mlp_layer_config,
            "message_config": self.message_layer_config,
            "decoder_layer_config": self.decoder_layer_config,
            "gpu": self.gpu,
            "device": self.device
        }

        return {**base_config, **mlp_config}

    def _init_graph(self):
        # Input layer
        self.input_layer: Dict[str, nn.Module] = dict()
        for layer_name, layer_config in self.input_layer_config.items():
            self.input_layer[layer_name] = MLPLayerV2(layer_config, prefix_name=f"{layer_name}_input")

        self.edge_layer = MLPLayerV2(
            self.message_passing_layer_config["edge_mlp_layer"], prefix_name="edge_input"
        )
        self.message_update_layer = MLPLayerV2(
            self.message_passing_layer_config["node_mlp_layer"], prefix_name="message"
        )

        # aggregator pooling
        agg_method = self.message_passing_layer_config["agg_method"]
        self.message_agg_pooling = globals()[agg_method](self.message_passing_layer_config["agg_layer"])

        # theta mlp
        self.theta_encode_mlp_layer = MLPLayerV2(self.theta_input_mlp_layer_config, prefix_name="theta_encode")

        # decoder MLPs
        decoder_layer_config = self.decoder_layer_config
        self.decoder_layer = nn.ModuleList(
            [
                MLPLayerV2(decoder_layer_config, prefix_name=f"decode_{i}")
                for i in range(decoder_layer_config["output_dim"])
            ]
        )

    @staticmethod
    def _edge_emb(node_emb: torch.Tensor, input_edge_indices: torch.Tensor) -> torch.Tensor:
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

    def message_passing_layer(self, x: [str, torch.Tensor]) -> torch.Tensor:
        input_laplace_coord: torch.Tensor = x["laplace_coord"]  # shape: (batch_size, node_num, coord_dim)
        input_node_fea: torch.Tensor = x["fiber_and_sheet"]  # shape: (batch_size, node_num, node_feature_dim)
        input_node_coord: torch.Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)
        input_edge_indices: torch.Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)

        input_selected_node: torch.Tensor = x["selected_node"]  # shape: (selected_node_num, )
        input_selected_node_num: torch.Tensor = x["selected_node_num"]  # shape: (1,)

        input_node_emb: torch.Tensor = torch.concat([input_laplace_coord, input_node_fea], dim=-1)

        # random select nodes
        batch_size, node_num, seq_num = input_edge_indices.shape

        select_batch = torch.arange(batch_size)

        select_indices = torch.randint(
            0, seq_num, (batch_size, input_selected_node_num, self.select_edge_num),
            dtype=torch.int64, device=self.device,
        )

        selected_edge = input_edge_indices[
            select_batch[:, None, None], input_selected_node[None, :, None], select_indices
        ]  # shape: (batch_size, selected_node_num, seq)

        # select node based on random results
        # shape: (batch_size, node_num, node_emb) => (batch_size, selected_node_num, node_emb) =>
        # (batch_size, selected_node_num, 1, node_emb) => (batch_size, selected_node_num, seq, node_emb)
        node_self_emb = input_node_emb[:, input_selected_node, :]
        node_self_seq_emb = node_self_emb.unsqueeze(dim=-2).expand(-1, -1, self.select_edge_num, -1)

        # # select edge based on random results
        # shape: (batch_size, selected_node_num, seq, node_emb)
        node_edge_seq_emb = self._edge_emb(input_node_emb, selected_edge)

        # shape: (batch_size, node_num, seq, coord) -> (batch_size, node_num, seq, edge_emb)
        edge_coord = self._edge_coord(input_node_coord, selected_edge, input_selected_node)
        edge_coord_seq_emb = self.edge_layer(edge_coord)

        emb_concat = torch.concat([node_self_seq_emb, node_edge_seq_emb, edge_coord_seq_emb], dim=-1)

        node_emb_up = self.message_update_layer(emb_concat)  # shape: (batch_size, node_num, seq, node_emb)

        node_emb_pooling = self.message_agg_pooling(node_emb_up)  # shape: (batch_size, node_num, node_emb)

        return node_self_emb + node_emb_pooling

    def forward(self, x: Dict[str, torch.Tensor]):
        # ====== Input data
        # ============ transform
        inputs: Dict[str, torch.Tensor] = {
            n: self.input_layer[n](t) for n, t in x.items() if n in self.input_layer
        }

        # # ============ input as is
        inputs["selected_node"] = x["selected_node"]
        inputs["selected_node_num"] = x["selected_node_num"]

        # ====== message passing layer: Encoder & Aggregate
        z_local = self.message_passing_layer(inputs)  # shape: (batch_size, node_num, node_emb)

        # encode global parameters theta
        mat_param: torch.Tensor = inputs["mat_param"]  # shape: (batch_size, mat_param)
        pressure: torch.Tensor = inputs["pressure"]  # shape: (batch_size, pressure)
        input_shape_coeffs: torch.Tensor = inputs["shape_coeffs"]  # shape: (batch_size, graph_feature)

        z_theta = self.theta_encode_mlp_layer(
            torch.concat([mat_param, pressure, input_shape_coeffs], dim=-1)
        )  # shape: (batch_size, theta_feature)

        # tile global values
        global_fea = z_theta.unsqueeze(dim=-2)  # shape: (batch_size, 1, emb)
        global_fea_expanded = torch.tile(global_fea, (1, z_local.shape[1], 1))  # shape: (batch_size, node_num, emb)

        encoding_emb = torch.concat((global_fea_expanded, z_local), dim=-1)  # shape: (batch_size, node_num, emb)

        # ====== Decoder:
        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [
            decode_mlp(encoding_emb) for decode_mlp in self.decoder_layer
        ]  # shape: List[(batch_size, node_num, 1)]

        # concatenate the predictions of each individual decoder mlp
        output = dict()

        output["displacement"] = torch.concat(individual_mlp_predictions, dim=-1)  # shape: (batch_size, node_num, 1)

        return output
