from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from common.constant import TRAIN_NAME
from pkg.train.layer.pooling_layer import MeanAggregator, SUMAggregator  # noqa
from pkg.train.model.base_model import BaseModule
from pkg.train.trainer.base_trainer import BaseTrainer
from pkg.utils.logs import init_logger
from task.passive_lv.data.datasets_train import FEHeartSageTrainDataset
from task.passive_lv.utils.module.mlp_layer_ln import MLPLayerLN

logger = init_logger("FE_PASSIVE_LV_HEART_SAGE")

torch.manual_seed(753)
torch.set_printoptions(precision=8)


class FEHeartSAGETrainer(BaseTrainer):
    """Trainer class for FEHeartSAGE model.

    Handles training and validation of the FEHeartSAGE model, including data preprocessing
    and metric computation.
    """

    dataset_class = FEHeartSageTrainDataset

    def __init__(self) -> None:
        """Initialize the trainer with dataset configuration."""
        super().__init__()

        # config relative to dataset
        dataset_config = self.dataset_class(self.task_data, TRAIN_NAME)

        self.displacement_mean = dataset_config.get_displacement_mean()
        self.displacement_std = dataset_config.get_displacement_std()

    def create_model(self) -> None:
        self.model = FEHeartSAGEModel(self.task_train)

    def validation_step_check(self, epoch: int, is_last_epoch: bool) -> bool:
        if epoch <= 20 or epoch % 5 == 0 or is_last_epoch:
            return True
        else:
            return False

    def compute_validation_loss(self, predictions: Dict[str, Tensor], labels: Dict[str, Tensor]):
        predictions["displacement"] = predictions["displacement"] * self.displacement_std + self.displacement_mean
        return self.compute_loss(predictions, labels)

    def compute_metrics(self, metrics_func: callable, predictions: Dict[str, Tensor], labels: Dict[str, Tensor]):
        predictions["displacement"] = predictions["displacement"] * self.displacement_std + self.displacement_mean
        return super().compute_metrics(metrics_func, predictions, labels)


class FEHeartSAGEModel(BaseModule):
    """A specialized Graph Neural Network model for heart left ventricle deformation prediction.

    This model is designed to predict displacement in finite element meshes of the left ventricle.
    The architecture combines geometric and feature information through:
    - Input encoders for node coordinates, features, and material parameters
    - Edge feature computation based on relative node positions
    - Message passing layers with either attention or MLP-based updates
    - Neighborhood information aggregation through mean/sum pooling
    - Global parameter integration (theta values and shape coefficients)
    - Dimension-specific MLP decoders for displacement prediction

    The model processes both local mesh topology and global material parameters
    to predict node-wise displacement values in the cardiac mesh.
    """

    def __init__(self, config: Dict, *args, **kwargs) -> None:
        """Initialize the FEHeartSAGE model.

        Args:
            config (Dict): Configuration dictionary containing model hyperparameters and
                         architecture specifications including:
                         - Number of edges to sample per node
                         - MLP configurations for node/edge/theta encoders
                         - Message passing layer configurations
                         - Decoder layer specifications
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(config, *args, **kwargs)

        # hyper-parameter config
        self.select_edge_num = config["select_edge_num"]

        # mlp layer config
        self.node_input_mlp_layer_config = config["node_input_mlp_layer"]
        self.edge_input_mlp_layer_config = config["edge_input_mlp_layer"]
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
            "edge_input_mlp_layer": self.edge_input_mlp_layer,
            "theta_input_mlp_layer": self.theta_input_mlp_layer_config,
            "message_config": self.message_layer_config,
            "decoder_layer_config": self.decoder_layer_config,
        }

        return {**base_config, **mlp_config}

    def _init_graph(self):
        """Initialize the graph neural network components.

        Sets up the following model components:
        - Node and edge encoder MLPs
        - Message aggregation pooling layer
        - Message passing node and edge update functions
        - Theta encoder MLP
        - Decoder MLPs for making predictions
        """
        # 2 encoder mlp
        self.node_encode_mlp_layer = MLPLayerLN(self.node_input_mlp_layer_config, prefix_name="node_encode")
        self.edge_encode_mlp_layer = MLPLayerLN(self.edge_input_mlp_layer_config, prefix_name="edge_encode")

        # aggregator pooling
        agg_method = self.message_passing_layer_config["agg_method"]
        self.message_agg_pooling = globals()[agg_method](self.message_passing_layer_config["agg_layer"])

        for i in range(self.message_layer_num):
            self.node_update_fn.append(
                MLPLayerLN(self.message_passing_layer_config["node_mlp_layer"], prefix_name=f"message_node_{i}")
            )
            self.edge_update_fn.append(
                MLPLayerLN(self.message_passing_layer_config["edge_mlp_layer"], prefix_name=f"message_edge_{i}")
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
        """Preprocess node features by concatenating node features with coordinates.

        Args:
            x (Dict[str, torch.Tensor]): Dictionary containing node features

        Returns:
            torch.Tensor: Concatenated node features and coordinates
        """
        input_node_fea: torch.Tensor = x["node_features"]  # shape: (batch_size, node_num, node_feature_dim)
        input_node_coord: torch.Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)

        return torch.concat([input_node_fea, input_node_coord], dim=-1)

    def _edge_emb(self, node_emb: torch.Tensor, input_edge_indices: torch.Tensor) -> torch.Tensor:
        """Create edge embeddings by gathering node embeddings based on edge indices.

        Args:
            node_emb (torch.Tensor): Node embedding tensor of shape (batch_size, node_num, emb_dim)
            input_edge_indices (torch.Tensor): Edge indices tensor of shape (batch_size, node_num, seq)

        Returns:
            torch.Tensor: Edge embeddings tensor
        """
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

    def _edge_coord(self, input_node_coord: torch.Tensor, input_edge_indices: torch.Tensor) -> torch.Tensor:
        """Create edge coordinate features from node coordinates and edge indices.

        Args:
            input_node_coord (torch.Tensor): Node coordinate tensor of shape (batch_size, node_num, coord_dim)
            input_edge_indices (torch.Tensor): Edge indices tensor of shape (batch_size, node_num, seq)

        Returns:
            torch.Tensor: Edge coordinate tensor containing relative coordinates and vertex coordinates
        """
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

        # combine node data + seq data => edge data
        # shape: (batch_size, node_num, seq, node_coord_dim) =>
        # (batch_size, node_num, seq, 2 * node_coord_dim)
        edge_vertex_coord: torch.Tensor = torch.concat([node_coord_expanded, node_seq_coord], dim=-1)
        edge_coord: torch.Tensor = node_coord_expanded - node_seq_coord

        return torch.concat([edge_coord, edge_vertex_coord], dim=-1)

    def random_select_nodes(self, indices: torch.Tensor) -> torch.Tensor:
        """Randomly select a subset of neighbor nodes for each node.

        Args:
            indices (torch.Tensor): Edge indices tensor of shape (batch_size, rows, cols)

        Returns:
            torch.Tensor: Randomly selected edge indices
        """
        batch_size, rows, cols = indices.shape

        random_indices = torch.randint(0, cols, (batch_size, rows, self.select_edge_num), dtype=torch.int64)

        batch_indices = torch.arange(batch_size)[:, None, None]

        row_indices = torch.arange(rows)[None, :, None]

        return indices[batch_indices, row_indices, random_indices]

    def message_passing_layer(self, x: Dict, node_emb: torch.Tensor) -> torch.Tensor:
        """Perform message passing between nodes in the graph.

        Args:
            x (Dict): Dictionary containing input tensors
            node_emb (torch.Tensor): Node embedding tensor

        Returns:
            torch.Tensor: Updated node embeddings after message passing
        """
        input_edge_indices: torch.Tensor = x["edges_indices"]  # shape: (batch_size, node_num, seq)
        input_node_coord: torch.Tensor = x["node_coord"]  # shape: (batch_size, node_num, coord_dim)

        for i in range(self.message_layer_num):
            selected_edge = self.random_select_nodes(input_edge_indices)  # shape: (batch_size, node_num, seq)

            edge_seq_emb = self._edge_emb(node_emb, selected_edge)  # shape: (batch_size, node_num, seq, node_emb)
            edge_coord = self._edge_coord(input_node_coord, selected_edge)  # shape: (batch_size, node_num, seq, coord)

            # shape: (batch_size, node_num, 1, node_emb) => (batch_size, node_num, seq, node_emb)
            node_emb_expanded = node_emb.unsqueeze(dim=-2).expand(-1, -1, selected_edge.shape[-1], -1)

            edge_seq_emb_up = torch.concat([node_emb_expanded, edge_seq_emb, edge_coord], dim=-1)

            edge_seq_emb_up = self.edge_update_fn[i](edge_seq_emb_up)  # shape: (batch_size, node_num, seq, edge_emb)

            edge_emb_pooling = self.message_agg_pooling(edge_seq_emb_up)  # shape: (batch_size, node_num, edge_emb)

            node_emb_up = torch.concat([node_emb, edge_emb_pooling], dim=-1)

            node_emb_up = self.node_update_fn[i](node_emb_up)  # shape: (batch_size, node_num, node_emb)

            node_emb = node_emb + node_emb_up

        return node_emb

    def forward(self, x: Dict[str, torch.Tensor]):
        # ====== Input data: (squeeze to align to previous project)
        input_theta = x["theta_vals"]  # shape: (batch_size, graph_feature)
        input_z_global = x["shape_coeffs"]  # shape: (batch_size, graph_feature)

        input_node = self._node_preprocess(x)  # shape: (batch_size, node_num, node_feature_dim+coord_dim)

        # ====== message passing layer: Encoder & Aggregate
        node_emb = self.node_encode_mlp_layer(input_node)  # shape: (batch_size, node_num, node_emb)

        z_local = self.message_passing_layer(x, node_emb)  # shape: (batch_size, node_num, node_emb)

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
        output = dict()
        output["displacement"] = torch.concat(individual_mlp_predictions, dim=-1)  # shape: (batch_size, node_num, 1)
        return output
