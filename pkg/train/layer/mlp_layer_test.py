import torch

from pkg.train.layer.mlp_layer import MLPLayerBase
from pkg.utils.logs import init_logger
from pkg.utils.model_summary import summary_model as summary

logger = init_logger("test_mlp_module")

if __name__ == "__main__":
    config = {
        "prefix_name": "test_mlp_module",
        "unit_sizes": [128, 64, 32, 16],
        "layer_name": "test",
        "layer_norm": True,
        "activation": "relu",
    }

    model = MLPLayerBase(config)
    logger.info(f"module config: {model.get_config()}")

    logger.info(summary(model, torch.rand((100, 128))))
