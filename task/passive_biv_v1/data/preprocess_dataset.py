import os
import sys

from pkg.utils import io
from task.passive_biv_v1.data.datasets import PassiveBiVPreprocessDataset
from common.constant import UNKNOWN_NAME

if __name__ == "__main__":
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = io.get_repo_path(cur_path)
    data_config = {
        "task_data_path": f"{repo_root_path}/pkg/data/passive_biv",
        "task_path": f"{repo_root_path}/task/passive_biv_v1",
        "gpu": False,
        "default_padding_value": -1,
        "exp_name": "v3_1",
        "chunk_file_size": 2,
    }

    data = PassiveBiVPreprocessDataset(data_config, UNKNOWN_NAME)

    data.preprocess()
