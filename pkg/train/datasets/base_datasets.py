import abc
import os
import platform
from typing import Dict, List, Optional, Set, Union
import sys
from pkg.utils.io import get_repo_path, load_yaml
from pkg.train.datasets import logger
from common.constant import TRAIN_NAME, MODEL_TRAIN

class BaseAbstractDataset(abc.ABC):
    """Abstract base class for dataset preparation and train.

    This class serves as a blueprint for datasets, providing common setup
    such as paths and hardware configuration. Subclasses are expected to
    implement methods for data processing and management.

    Attributes:
    ----------
    base_data_path : str
        Path to the base data directory.
    base_task_path : str
        Path to the base task directory.
    gpu : bool
        Flag indicating if GPU should be used.
    cuda_core : str
        Identifier for the CUDA core to be used.
    data_type : str
        Type of the data being handled (e.g., 'train', 'test').
    exp_name : str, optional
        Name of the experiment, if specified.
    """

    def __init__(self, data_config: Dict, data_type: str, process: Optional[str] = None, *args, **kwargs) -> None:
        """Initialize the dataset with configuration details.

        Parameters:
        ----------
        data_config : Dict
            Dictionary containing configuration details such as paths and hardware setup.
        data_type : str
            String specifying the type of data (e.g., 'train', 'test').
        process: str
            String specifying the process of dataset
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.
        """
        logger.info(f"====== init {data_type} data config ======")
        logger.info(data_config)

        # common config
        # === Hardware configuration
        self.gpu = data_config["gpu"]
        self.cuda_core = data_config.get("cuda_core", "gpu:0")
        self.platform = platform.system()

        # === data type
        self.task_name = data_config["task_name"]
        self.data_type = data_type

        # === exp
        self.exp_name = data_config["exp_name"]

        # common path
        # === base path
        self.base_repo_path = data_config["repo_path"]
        self.base_data_path = data_config["task_data_path"]
        self.base_task_path = data_config["task_path"]

        if not os.path.isdir(self.base_data_path):
            raise NotADirectoryError(f"No directory at: {self.base_data_path}")

        # === traditional model training dataset path (non-tfrecord version)
        self.stats_data_path = f"{self.base_data_path}/stats/{self.data_type}"
        if process == MODEL_TRAIN:  # if process = training, we will load train stats for validation dataset
            self.stats_data_path = f"{self.base_data_path}/stats/{TRAIN_NAME}"
        self.dataset_path = f"{self.base_data_path}/datasets/{self.data_type}"

        logger.info(f"base_data_path is {self.base_data_path}")
        logger.info(f"base_task_path is {self.base_task_path}")
        logger.info(f"stats_data_path is {self.stats_data_path}")
        logger.info(f"dataset_path is {self.dataset_path}")

        self.data_size_path = f"{self.base_data_path}/stats/{self.data_type}/{self.data_type}_data_size.npy"
        logger.info(f"data_size_path is {self.data_size_path}")

        # hdf5 config
        self.dataset_h5_path = f"{self.dataset_path}" + "/data_{}.h5"

        # tfrecord config
        # === tfrecord model training dataset path (tfrecord version)
        self.tfrecord_data_path = f"{self.dataset_path}" + "/data_{}.tfrecord"

        logger.info(f"tfrecord_data_path is {self.tfrecord_data_path}")

        # === tfrecord file compression type
        self.compression_type = None

        # tfrecord features
        self.context_description: Optional[Dict[str, str]] = None  # please overwrite this variable

        self.feature_description: Optional[Dict[str, str]] = None  # please overwrite this variable


class BaseAbstractDataPreparationDataset(abc.ABC):
    """Abstract base class for data preparation tasks.

    This class defines the blueprint for preparing datasets, generating data,
    and computing data statistics. Subclasses should implement the methods
    defined here.

    Methods:
    --------
    prepare_dataset_process():
        Prepare the dataset. Must be implemented by subclasses.
    _data_generation():
        Generate data. Must be implemented by subclasses.
    _data_stats():
        Compute data statistics. Must be implemented by subclasses.
    """

    @abc.abstractmethod
    def prepare_dataset_process(self):
        """Prepare the dataset.

        This method should be overridden in subclasses to define the
        specific steps needed to prepare the dataset.
        """
        raise NotImplementedError("Subclasses must implement the prepare_dataset_process method.")

    @abc.abstractmethod
    def _data_generation(self):
        """Generate data.

        This method should be overridden in subclasses to define the
        specific steps needed to generate data.
        """
        raise NotImplementedError("Subclasses must implement the _data_generation method.")

    @abc.abstractmethod
    def _data_stats(self):
        """Compute data statistics.

        This method should be overridden in subclasses to define how
        data statistics should be computed.
        """
        raise NotImplementedError("Subclasses must implement the _data_stats method.")


class BaseAbstractTrainDataset(BaseAbstractDataset):
    @abc.abstractmethod
    def get_head_inputs(self, batch_size: int) -> Dict:
        """Generate and return the inputs required for the model head.

        This method must be implemented by subclasses to generate the necessary
        input data for the model's head, based on the provided batch size. It mainly used for
        printing model architecture.

        Parameters:
        ----------
        batch_size : int
            The number of samples to generate inputs for.

        Returns:
        ----------
        Dict : A dictionary containing the inputs for the model head.
        """
        raise NotImplementedError("Subclasses must implement get_head_inputs method")

    @abc.abstractmethod
    def __len__(self):
        """
        Return the size of the dataset.

        Subclasses must override this method to provide the logic for
        determining the size of the dataset.

        Returns:
        ----------
        int : The number of items in the dataset.
        """
        raise NotImplementedError("Subclasses must implement __len__ method")


def import_data_config(task_name: str, model_name: str, dataset_name: str) -> Dict:
    # generate root path
    cur_path = os.path.abspath(sys.argv[0])

    repo_root_path = get_repo_path(cur_path)

    # fetch data config
    base_config = load_yaml(f"{repo_root_path}/task/{task_name}/{model_name}/config/train_config.yaml")
    data_config = base_config["task_data"]

    task_base = base_config["task_base"]
    data_config["task_name"] = task_base["task_name"]
    data_config["exp_name"] = task_base["exp_name"]

    data_config["repo_path"] = repo_root_path
    data_config["task_data_path"] = f"{repo_root_path}/pkg/data/{dataset_name}"
    data_config["task_path"] = f"{repo_root_path}/task/{task_name}/{model_name}"
    data_config["gpu"] = base_config["task_base"]["gpu"]
    data_config["exp_name"] = base_config["task_base"]["exp_name"]

    return data_config
