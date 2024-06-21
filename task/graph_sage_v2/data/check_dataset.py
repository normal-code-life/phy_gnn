import os
import sys

from pkg.utils import io
from task.graph_sage_v2.data.datasets import GraphSagePreprocessDataset, GraphSageTrainDataset

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)
    data_config = {
        "task_data_path": f"{repo_root_path}/pkg/data/lvData",
        "task_path": f"{repo_root_path}/task/graph_sage",
        "gpu": False,
        "n_shape_coeff": 32,
        "edge_indices_generate_method": 0,
        "default_padding_value": -1,
        "exp_name": "10",
        "chunk_size": 20
    }

    data_preprocess = GraphSagePreprocessDataset(data_config, "validation")

    data_preprocess.preprocess()

    # dataset = GraphSageTrainDataset(data_config, "validation")
    #
    # res = dataset.get_head_inputs(10)
    #
    # for k in res:
    #     print(res[k].shape)
    #
    # print(res["edges_indices"][:, 0: 5, 0:10])


    # for sample, labels in dataset:
    #     if batch > 1:
    #         break
    #
    #     print(sample)
    #     print(labels)
    #
    #     batch += 1
    #
