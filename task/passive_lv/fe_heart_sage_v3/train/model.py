from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor

from common.constant import TRAIN_NAME
from pkg.train.layer.pooling_layer import MeanAggregator, SUMAggregator  # noqa
from pkg.train.model.base_model import BaseModule
from pkg.train.trainer.base_trainer import BaseTrainer
from pkg.utils.logs import init_logger
from task.passive_lv.data.datasets_train import FEHeartSageTrainDataset
from task.passive_lv.utils.module.mlp_layer_ln import MLPLayerV2

logger = init_logger("FEPassiveLVHeartSage")

torch.manual_seed(753)
torch.set_printoptions(precision=8)


class FEHeartSageV3Trainer(BaseTrainer):
    """Trainer class for FEHeartSAGE model.

    Handles training and validation of the FEHeartSAGE model, including data preprocessing,
    node sampling, and metric computation.
    """

    dataset_class = FEHeartSageTrainDataset

    def __init__(self) -> None:
        """Initialize the trainer with dataset configuration and device setup."""
        super().__init__()

        self.selected_node_num = self.task_train["select_node_num"]

        self.device = "cuda" if self.gpu else "cpu"

        # config relative to dataset
        dataset_config = self.dataset_class(self.task_data, TRAIN_NAME)

        self.displacement_mean = dataset_config.get_displacement_mean()
        self.displacement_std = dataset_config.get_displacement_std()

    def create_model(self) -> None:
        self.model = FEHeartSAGEModel(self.task_train)

    def compute_validation_loss(self, predictions: Dict[str, Tensor], labels: Dict[str, Tensor]):
        predictions["displacement"] = predictions["displacement"] * self.displacement_std + self.displacement_mean
        return self.compute_loss(predictions, labels)

    def compute_metrics(self, metrics_func: callable, predictions: Dict[str, Tensor], labels: Dict[str, Tensor]):
        predictions["displacement"] = predictions["displacement"] * self.displacement_std + self.displacement_mean
        return super().compute_metrics(metrics_func, predictions, labels)

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
    """Graph Neural Network model.

    This model is designed to predict displacement in finite element meshes of the left ventricle
    by learning node embeddings in graphs.
    It performs message passing between nodes to aggregate neighborhood information and make predictions.
    The model consists of:
    - Input encoders for nodes, edges and global features
    - Message passing layers with attention or MLP architectures
    - Aggregation functions to combine neighbor messages
    - A decoder to make final displacement predictions
    """

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        """Initialize the FEHeartSAGE model.

        Args:
            config: Configuration dictionary containing model hyperparameters and
                   architecture specifications
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(config, *args, **kwargs)

        # hyper-parameter config
        self.select_edge_num = config["select_edge_num"]
        self.default_padding_value = config.get("default_padding_value", -1)

        # mlp layer config
        self.input_layer_config = config["input_layer"]
        self.edge_mlp_layer_config = config["edge_mlp_layer"]
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
        """Initialize the graph neural network components.

        Sets up the following model components:
        - Input layers for feature preprocessing
        - Edge and edge Laplacian MLPs
        - Message passing layers (attention or MLP based)
        - Message aggregation pooling
        - Global parameter encoder
        - Decoder MLPs
        """
        # Input layer
        self.input_layer: nn.ModuleList = nn.ModuleList()
        for layer_name, layer_config in self.input_layer_config.items():
            self.input_layer.append(MLPLayerV2(layer_config, prefix_name=layer_name))

        self.edge_mlp_layer = MLPLayerV2(self.edge_mlp_layer_config, prefix_name="edge_input")

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
        """Randomly select edges for each node.

        Args:
            indices: Edge indices tensor
            device: Device to place tensors on
            selected_edge_num: Number of edges to select per node

        Returns:
            Tensor of selected edge indices
        """
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
        """Generate edge embeddings from node embeddings and edge indices.

        Args:
            node_emb: Node embedding tensor
            input_edge_indices: Edge indices tensor

        Returns:
            Edge embedding tensor
        """
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
        """Generate edge coordinate features from node coordinates and edge indices.

        Args:
            input_node_coord: Node coordinate tensor
            input_edge_indices: Edge indices tensor

        Returns:
            Edge coordinate tensor
        """
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

        input_node_coord_emb: Tensor = x_trans["node_coord_emb"]  # shape: (batch_size, node_num, coord_dim)
        input_node_fea_emb: Tensor = x_trans["node_features_emb"]  # shape: (batch_size, node_num, node_feature_dim

        input_edge_indices: Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)
        selected_node: Tensor = x["selected_node"][0]  # shape: (batch_size, selected_node_num)

        theta_vals_emb: Tensor = x_trans["theta_vals_emb"]  # shape: (batch_size, mat_param)
        input_shape_coeffs_emb: Tensor = x_trans["shape_coeffs_emb"]  # shape: (batch_size, graph_feature)

        # ====== Message passing Encoder & Aggregate
        # ============ generate node emb (node emb itself)  TODO: test whether to involve the node itself
        input_node_emb = input_node_coord_emb + input_node_fea_emb  # (batch_size, node_num, node_emb)
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

        edge_coord_emb = self.edge_mlp_layer(edge_coord)  # (batch_size, node_num, seq, coord_emb)

        # ============ agg node, edge, coord emb and send to message passing layer & pooling
        emb_concat = torch.concat([node_seq_emb, edge_seq_emb, edge_coord_emb], dim=-1)[:, selected_node, :, :]

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
            torch.concat([theta_vals_emb, input_shape_coeffs_emb], dim=-1)
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
