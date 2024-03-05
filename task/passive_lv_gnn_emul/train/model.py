from pprint import pformat
from pkg.train.trainer.base_trainer import TrainerConfig, BaseTrainer
from pkg.train.model.base_model import BaseModuleConfig, BaseModule
from pkg.utils.logging import init_logger
from pkg.utils import io
from task.passive_lv_gnn_emul.train.datasets import LvDataset
from common.constant import TRAIN_NAME, VALIDATION_NAME, TEST_NAME
import os
import sys
from typing import Dict, Sequence
# from torchsummary import summary
from pkg.train.module.mlp_layer import MLPConfig, MLPModule
import torch.nn as nn
import torch
from task.passive_lv_gnn_emul.train.message_passing_layer import MessagePassingConfig, MessagePassingModule
from pkg.tf_utils.method import segment_sum

logger = init_logger("PassiveLvGNNEmul")


class PassiveLvGnnEmulConfig(TrainerConfig):
    def __init__(self, config_path: str) -> None:

        super().__init__(config_path)
        logger.info(f"====== config init ====== \n{pformat(self.config)}")


class PassiveLvGNNEmulTrainer(BaseTrainer):
    def __init__(self, config: PassiveLvGnnEmulConfig) -> None:
        super().__init__(config)

        trainer_param = self.task_trainer

        logger.info(f"Data path: {self.task_data['task_data_path']}")
        logger.info(f'Training epochs: {trainer_param["step_param"]["epochs"]}')
        logger.info(f'Learning rate: {trainer_param["optimizer_param"]["learning_rate"]}')
        logger.info(f'Fixed LV geom: {trainer_param["fixed_geom"]}\n')

    def read_dataset(self):
        task_data = self.task_data

        train_data = LvDataset(task_data, TRAIN_NAME)
        logger.info(f"Number of train data points: {len(train_data)}")

        validation_data = LvDataset(task_data, VALIDATION_NAME)
        logger.info(f"Number of validation_data data points: {len(validation_data)}")

        return train_data, validation_data

    def create_model(
        self, senders: Sequence[int], receivers: Sequence[int], real_node_indices: Sequence[bool], n_total_nodes: int
    ):
        config = PassiveLvGNNEmulConfig(self.task_train, senders, receivers, real_node_indices, n_total_nodes)
        model = PassiveLvGNNEmulModel(config)

        def print_model(model):
            # logger.info(model)
            for name, module in model.named_children():
                logger.info(f"Submodule: {name}")
                logger.info(module)
                if isinstance(module, nn.Module):
                    print_model(module)

        print_model(model)

        return model

    def fit(self):
        train_data, validation_data = self.read_dataset()

        senders = train_data.get_senders()
        receivers = train_data.get_receivers()
        real_node_indices = train_data.get_real_node_indices()
        n_total_nodes = train_data.get_n_total_nodes()

        model = self.create_model(senders, receivers, real_node_indices, n_total_nodes)

        optimizer_param = self.task_trainer["optimizer_param"]

        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_param["learning_rate"])


class PassiveLvGNNEmulConfig(BaseModuleConfig):
    def __init__(
        self,
        config: Dict,
        senders: Sequence[int],
        receivers: Sequence[int],
        real_node_indices: Sequence[int],
        n_total_nodes: int,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)

        # mlp layer config
        self.input_mlp_layer_config = config["input_mlp_layer"]
        self.message_passing_layer_config = config["message_passing_layer"]
        self.decoder_layer_config = config["decoder_layer"]

        # message passing config
        self.message_passing_layer_config["senders"] = senders
        self.message_passing_layer_config["receivers"] = receivers
        self.message_passing_layer_config["n_total_nodes"] = n_total_nodes

        # other config
        self.real_node_indices = real_node_indices

        logger.info(f'Message passing steps: {config["message_passing_steps"]}')
        logger.info(f'Num. shape coeffs: {config["n_shape_coeff"]}')

    def get_config(self):
        base_config = super().get_config()

        mlp_config = {
            "input_mlp_layer_config": self.input_mlp_layer_config,
            "message_passing_layer_config": self.message_passing_layer_config,
            "decoder_layer_config": self.decoder_layer_config,
            "real_node_indices": self.real_node_indices,
        }

        return {**base_config, **mlp_config}


class PassiveLvGNNEmulModel(BaseModule):
    def __init__(self, config: PassiveLvGNNEmulConfig) -> None:
        super().__init__(config)

        self._init_graph(config)

    def _init_graph(self, config: PassiveLvGNNEmulConfig):
        # 3 encoder mlp
        input_mlp_config = config.input_mlp_layer_config

        node_encode_mlp_config = MLPConfig(input_mlp_config, prefix_name="node_encode")
        self.node_encode_mlp_layer = MLPModule(node_encode_mlp_config)

        edge_encode_mlp_config = MLPConfig(input_mlp_config, prefix_name="edge_encode")
        self.edge_encode_mlp_layer = MLPModule(edge_encode_mlp_config)

        # theta mlp
        theta_encode_mlp_config = MLPConfig(input_mlp_config, prefix_name="theta_encode")
        self.theta_encode_mlp_layer = MLPModule(theta_encode_mlp_config)

        # decoder MLPs
        decoder_layer_config = config.decoder_layer_config
        decoder_mlp_config = MLPConfig(decoder_layer_config["mlp_layer"], prefix_name="decode")
        self.decoder_layer = [MLPModule(decoder_mlp_config) for _ in range(decoder_layer_config["output_dim"])]

        # 2K processor mlp
        message_passing_layer_config = MessagePassingConfig(config.message_passing_layer_config)
        self.message_passing_layer = MessagePassingModule(message_passing_layer_config)

    def forward(self, x):
        input_node, input_edge, input_theta, input_z_global, _ = x  # shape: (126, 1), (440, 3), (1, 2), (2,)

        # ====== Encoder:
        # encode vertices and edges
        node = self.node_encode_mlp_layer(input_node)  # shape: (126, 40)
        edge = self.edge_encode_mlp_layer(input_edge)  # shape: (440, 40)

        # perform K rounds of message passing
        node, edge = self.message_passing_layer(node, edge)  # shape: (126, 40), (440, 40)

        # aggregate incoming messages to each node
        incoming_message = segment_sum(edge, self.config.receivers, self.config.n_total_nodes)  # shape: (126, 40)

        # final local learned representation is a concatenation of vector embedding and incoming messages
        z_local = torch.concat((node, incoming_message), dim=-1)  # shape: (126, 80)

        # only need local representation for real nodes
        z_local = z_local[self.real_node_indices]  # shape: (96, 80)

        # encode global parameters theta
        z_theta = self.theta_encode_mlp_layer(input_theta)  # shape: (1, 2) => (1, 40)

        # tile global values (z_theta and optionally z_global) to each individual real node
        if input_z_global is None:
            globals_array = torch.tile(z_theta, (z_local.shape[0], 1))  # shape: (96, 40)
        else:
            # stack z_global with z_theta if z_global is inputted
            z_global_reshape = input_z_global.reshape(1, -1)  # shape: (2, ) => (1, 2)
            global_embedding = torch.hstack((z_theta, z_global_reshape))  # shape: (1, 40) + (1, 2) => (1, 42)
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
    config_path = f"{task_dir}/train_config.yaml"

    lv_config = PassiveLvGnnEmulConfig(config_path)
    model = PassiveLvGNNEmulTrainer(lv_config)
    model.fit()
