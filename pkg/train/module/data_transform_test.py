import numpy as np
import torch

from pkg.train.module.data_transform import MaxMinNorm


def test_maxminnorm():
    # 模拟 normalization_config
    data_path = "./feature.npz"
    config = {"feature1": data_path, "feature2": data_path}

    # 假设 npz 文件存储为 numpy 数组，并模拟加载
    np.savez(data_path, max_val=np.array([10, 5, 4]), min_val=np.array([1, 2, 3]))

    # 实例化 MaxMinNorm
    normalizer = MaxMinNorm(config, global_scaling=False, coarse_dim=True)

    # 测试样本
    context = {
        "feature1": torch.from_numpy(np.arange(12).reshape(4, 3)),
    }
    feature = {
        "feature2": torch.from_numpy(np.arange(12, 24).reshape(4, 3)),
    }

    normalized_context, normalized_feature = normalizer((context, feature))

    print(f"normalized_context: {normalized_context}")
    print(f"normalized_feature: {normalized_feature}")


if __name__ == "__main__":
    test_maxminnorm()
