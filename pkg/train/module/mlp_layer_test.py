from pkg.train.module.mlp_layer import MLPConfig, MLPModule
from torchsummary import summary
from pkg.utils.logging import init_logger

logger = init_logger("test_mlp_module")

if __name__ == "__main__":
    config = {
        "prefix_name": "test_mlp_module",
        "unit_sizes": [128, 64, 32],
        "layer_name": "test",
        "layer_norm": True,
        "activation": "relu",
    }

    config = MLPConfig(config)
    logger.info(config.get_config())

    model = MLPModule(config)
    logger.info(model)

    # 使用 torchsummary 打印模型结构
    logger.info(summary(model, (128,)))
