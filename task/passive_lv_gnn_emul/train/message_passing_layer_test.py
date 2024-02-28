from task.passive_lv_gnn_emul.train.message_passing_layer import MessagePassingConfig, MessagePassingModule
from pkg.utils.logging import init_logger

# from torchsummary import summary


logger = init_logger("test_message_passing_module")

if __name__ == "__main__":
    config = {
        "prefix_name": "test_message_passing_module",
        "mlp_layer": {"unit_sizes": [128, 64, 32], "layer_name": "test", "layer_norm": True, "activation": "relu"},
        "K": 2,
        "senders": [0, 1, 2, 3],
        "receivers": [2, 3, 4, 5],
        "n_total_nodes": 6,
    }

    config = MessagePassingConfig(config)
    logger.info(config.get_config())

    model = MessagePassingModule(config)
    logger.info(model)

    # 使用 torchsummary 打印模型结构
    # logger.info(summary(model, [(128, 40), (128, 40)]))
