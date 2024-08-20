from torchsummary import summary

from pkg.utils.logs import init_logger
from task.passive_biv.utils.module.mlp_layer_ln import MLPLayerLN

logger = init_logger("test_mlp_module")

if __name__ == "__main__":
    config = {
        "prefix_name": "test_mlp_module",
        "unit_sizes": [128, 64, 32, 16],
        "layer_name": "test",
        "layer_norm": True,
        "activation": "relu",
    }

    model = MLPLayerLN(config)
    logger.info(f"module config: {model.get_config()}")

    # 使用 torchsummary 打印模型结构
    logger.info(summary(model, (128,)))
