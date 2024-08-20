import os
import sys

from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader

from pkg.utils import io  # Importing a custom utility module for getting the repository path


def measure_io_speed(directory_path):
    # Check if the provided path is a valid directory
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    # Walk through the directory and all subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Get the full path of the current file
            file_path = os.path.join(root, file)

            dataset = TFRecordDataset(str(file_path), None, context_description, None, None, sequence_description, None)
            data_loader = DataLoader(dataset, batch_size=1, num_workers=0)

            for context, fea in data_loader:
                print(len(fea["node_coord"]))


if __name__ == "__main__":
    context_description = {
        "index": "int",
        "points": "int",
    }

    sequence_description = {
        "node_coord": "float",
        "laplace_coord": "float",
        "fiber_and_sheet": "float",
        "edges_indices": "int",
        "shape_coeffs": "float",
        "mat_param": "float",
        "pressure": "float",
        "displacement": "float",
        "stress": "float",
    }

    # Get the absolute path of the current script
    cur_path = os.path.abspath(sys.argv[0])

    # Get the repository root path using a custom utility function
    repo_dir = io.get_repo_path(cur_path)

    # Define the directory to test the I/O speed
    directory_to_test = f"{repo_dir}/pkg/data/passive_biv/datasets/train"

    # Call the function to measure the I/O speed in the specified directory
    measure_io_speed(directory_to_test)
