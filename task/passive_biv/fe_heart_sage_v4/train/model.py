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

logger = init_logger("FEPassiveBivHeartSage")

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

    def post_transform_data(
        self, data: (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor])
    ) -> (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor]):
        inputs, labels = super().post_transform_data(data)

        batch_size, node_num, _ = inputs["edges_indices"].shape

        selected_node = torch.randint(
            0, node_num, size=(self.selected_node_num,), dtype=torch.int64, device=self.device
        )

        inputs["selected_node"] = selected_node.unsqueeze(0).expand(batch_size, -1)

        for label_name in self.labels:
            labels[label_name] = labels[label_name][:, selected_node, :]

        return inputs, labels

    def post_transform_val_data(
        self, data: (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor])
    ) -> (Union[Dict[str, Tensor], Tensor], Union[Dict[str, Tensor], Tensor]):
        inputs, labels = super().post_transform_val_data(data)

        batch_size, node_num, _ = inputs["edges_indices"].shape

        selected_node = torch.arange(node_num, device=self.device).unsqueeze(0).expand(batch_size, -1)

        inputs["selected_node"] = selected_node

        return inputs, labels


class FEHeartSAGEModel(BaseModule):
    """https://github.com/raunakkmr/GraphSAGE."""

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        # hyper-parameter config
        self.select_edge_num = config["select_edge_num"]
        self.default_padding_value = config.get("default_padding_value", -1)

        # mlp layer config
        self.input_layer_config = config["input_layer"]
        self.edge_mlp_layer_config = config["edge_mlp_layer"]
        self.edge_laplace_mlp_layer_config = config["edge_laplace_mlp_layer"]
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
            "device": self.device,
        }

        return {**base_config, **mlp_config}

    def _init_graph(self):
        # Input layer
        self.input_layer: nn.ModuleList = nn.ModuleList()
        for layer_name, layer_config in self.input_layer_config.items():
            self.input_layer.append(MLPLayerV2(layer_config, prefix_name=layer_name))

        self.edge_mlp_layer = MLPLayerV2(self.edge_mlp_layer_config, prefix_name="edge_input")

        self.edge_laplace_mlp_layer = MLPLayerV2(self.edge_laplace_mlp_layer_config, prefix_name="edge_laplace_input")

        if self.message_passing_layer_config["arch"] == "attention":
            self.message_update_layer = nn.TransformerEncoderLayer(
                d_model=self.message_passing_layer_config["message_update_layer"].get("d_model", 128),
                nhead=self.message_passing_layer_config["message_update_layer"].get("nhead", 4),
                dim_feedforward=self.message_passing_layer_config["message_update_layer"].get("dim_feedforward", 512),
                dropout=self.message_passing_layer_config["message_update_layer"].get("dropout", 0.1),
                # device=self.device,
                batch_first=True,
            )
            self.message_update_layer_mlp = MLPLayerV2(
                self.message_passing_layer_config["message_update_layer_mlp"], prefix_name="message"
            )
        elif self.message_passing_layer_config["arch"] == "mlp":
            self.message_update_layer = MLPLayerV2(
                self.message_passing_layer_config["message_update_layer"], prefix_name="message"
            )
        else:
            raise ValueError(f"please define the arch properly, current is {self.message_passing_layer_config['arch']}")

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
    def _random_select_edge(indices: Tensor, device: str, selected_edge_num: int) -> Tensor:
        batch_size, node_num, seq_num = indices.shape

        select_batch = torch.arange(batch_size, device=device)

        select_indices = torch.randint(
            0,
            seq_num,
            (batch_size, node_num, selected_edge_num),
            dtype=torch.int64,
            device=device,
        )  # TODO: train/test setup different edge num

        selected_node = torch.arange(node_num, device=device)

        return indices[
            select_batch[:, None, None], selected_node[None, :, None], select_indices
        ]  # shape: (batch_size, selected_node_num, seq)

    @staticmethod
    def _generate_edge_emb(node_emb: Tensor, input_edge_indices: Tensor) -> Tensor:
        emb_dim: int = node_emb.shape[-1]  # feature for each of the node
        seq: int = input_edge_indices.shape[-1]  # neighbours seq for each of the center node

        # === expand node feature to match indices shape
        # shape: (batch_size, node_num, emb) =>
        # (batch_size, node_num, 1, emb) =>
        # (batch_size, node_num, seq, emb)
        node_emb_expanded: Tensor = node_emb.unsqueeze(2).expand(-1, -1, seq, -1)

        # parse seq data
        # === expand indices to match feature shape
        # shape: (batch_size, node_num, seq) =>
        # (batch_size, node_num, seq, 1) =>
        # (batch_size, node_num, seq, emb)
        edge_seq_indices: Tensor = input_edge_indices.unsqueeze(-1).expand(-1, -1, -1, emb_dim)

        # === gather feature/coord
        return torch.gather(node_emb_expanded, 1, edge_seq_indices)

    @staticmethod
    def _generate_edge_coord(input_node_coord: Tensor, input_edge_indices: Tensor) -> torch.Tensor:
        coord_dim: int = input_node_coord.shape[-1]  # coord for each of the node
        seq: int = input_edge_indices.shape[-1]  # neighbours seq for each of the center node

        # === expand node feature to match indices shape
        # shape: (batch_size, node_num, node_coord_dim) =>
        # (batch_size, node_num, 1, node_coord_dim) =>
        # (batch_size, node_num, seq, node_coord_dim)
        node_coord_expanded: Tensor = input_node_coord.unsqueeze(2).expand(-1, -1, seq, -1)

        # parse seq data
        # === expand indices to match feature shape
        # shape: (batch_size, node_num, seq) =>
        # (batch_size, node_num, seq, 1) =>
        # (batch_size, node_num, seq, node_coord_dim)
        indices_coord_expanded: Tensor = input_edge_indices.unsqueeze(-1).expand(-1, -1, -1, coord_dim)

        # === gather coord
        node_seq_coord: Tensor = torch.gather(node_coord_expanded, 1, indices_coord_expanded)

        edge_coord: Tensor = node_coord_expanded - node_seq_coord

        return edge_coord

    def forward(self, x: Dict[str, Tensor]):
        # ====== Input data
        # ============ input transform
        x_trans: Dict[str, Tensor] = {}
        for preprocess_layer in self.input_layer:
            n = preprocess_layer.get_prefix_name
            if n in self.input_layer_config:
                x_trans[f"{n}_emb"] = preprocess_layer(x[n])

        # ============ input fetch
        input_node_coord: Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)
        input_node_laplace_coord: Tensor = x["laplace_coord"]  # shape: (batch_size, node_num, coord_dim)
        input_node_laplace_coord_emb: Tensor = x_trans["laplace_coord_emb"]  # shape: (batch_size, node_num, coord_dim)
        input_node_fea_emb: Tensor = x_trans["fiber_and_sheet_emb"]  # shape: (batch_size, node_num, node_feature_dim

        input_edge_indices: Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)
        selected_node: Tensor = x["selected_node"][0]  # shape: (batch_size, selected_node_num)

        input_mat_param_emb: Tensor = x_trans["mat_param_emb"]  # shape: (batch_size, mat_param)
        input_pressure_emb: Tensor = x_trans["pressure_emb"]  # shape: (batch_size, pressure)
        input_shape_coeffs_emb: Tensor = x_trans["shape_coeffs_emb"]  # shape: (batch_size, graph_feature)

        # ====== Message passing Encoder & Aggregate
        # ============ generate node emb (node emb itself)  TODO: test whether to involve the node itself
        input_node_emb = input_node_laplace_coord_emb + input_node_fea_emb  # (batch_size, node_num, node_emb)
        node_seq_emb = input_node_emb.unsqueeze(dim=-2).expand(
            -1, -1, self.select_edge_num, -1
        )  # (batch_size, node_num, 1, node_emb) => (batch_size, node_num, seq, node_emb)

        # ============ generate edge emb (agg by neighbours emb)
        selected_edge = self._random_select_edge(
            input_edge_indices, self.device, self.select_edge_num
        )  # shape: (batch_size, node_num, seq(of edge))
        edge_seq_emb = self._generate_edge_emb(
            input_node_emb, selected_edge
        )  # shape: (batch_size, node_num, seq, node_emb)

        # ============ generate relative coord emb (agg vertices emb at both ends + segment emb)
        edge_coord = self._generate_edge_coord(
            input_node_coord, selected_edge
        )  # (batch_size, node_num, seq, coord_emb)
        edge_laplace_coord = self._generate_edge_coord(
            input_node_laplace_coord, selected_edge
        )  # (batch_size, node_num, seq, coord_emb)

        # node_coord_emb = self.node_mlp_layer(node_coord_expanded)  # (batch_size, node_num, seq, coord_emb)
        # node_seq_coord_emb = self.node_mlp_layer(node_seq_coord)  # (batch_size, node_num, seq, coord_emb)
        edge_coord_emb = self.edge_mlp_layer(edge_coord)  # (batch_size, node_num, seq, coord_emb)
        edge_laplace_coord_emb = self.edge_laplace_mlp_layer(
            edge_laplace_coord
        )  # (batch_size, node_num, seq, coord_emb)

        coord_emb = edge_coord_emb + edge_laplace_coord_emb

        # ============ agg node, edge, coord emb and send to message passing layer & pooling
        emb_concat = torch.concat([node_seq_emb, edge_seq_emb, coord_emb], dim=-1)[:, selected_node, :, :]

        if self.message_passing_layer_config["arch"] == "attention":
            node_emb_up = emb_concat.view(
                -1, emb_concat.shape[2], emb_concat.shape[3]
            )  # shape: (batch_size * selected_node_num, seq_len, embed_dim)
            node_emb_up = self.message_update_layer(node_emb_up)  # shape: (batch_size * node_num, seq, node_emb)
            node_emb_up = node_emb_up.view(emb_concat.shape)  # shape: (batch_size, node_num, seq, node_emb)
            node_emb_up = self.message_update_layer_mlp(node_emb_up)
        else:
            node_emb_up = self.message_update_layer(emb_concat)  # shape: (batch_size, node_num, seq, node_emb)

        node_emb_pooling = self.message_agg_pooling(node_emb_up)  # shape: (batch_size, node_num, node_emb)

        # ============ res
        z_local = input_node_emb[:, selected_node, :] + node_emb_pooling

        # ====== Encode global parameters theta
        global_fea = self.theta_encode_mlp_layer(
            torch.concat([input_mat_param_emb, input_pressure_emb, input_shape_coeffs_emb], dim=-1)
        )  # shape: (batch_size, theta_feature)
        global_fea = global_fea.unsqueeze(dim=-2)  # shape: (batch_size, 1, emb)
        global_fea_expanded = torch.tile(global_fea, (1, z_local.shape[1], 1))  # shape: (batch_size, node_num, emb)

        # ====== concat local & global
        encoding_emb = torch.concat((global_fea_expanded, z_local), dim=-1)  # shape: (batch_size, node_num, emb)

        # ====== decoder
        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [
            decode_mlp(encoding_emb) for decode_mlp in self.decoder_layer
        ]  # shape: List[(batch_size, node_num, 1)]

        # concatenate the predictions of each individual decoder mlp
        output = dict()

        output["displacement"] = torch.concat(individual_mlp_predictions, dim=-1)  # shape: (batch_size, node_num, 1)

        return output
