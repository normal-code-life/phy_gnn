from typing import List, Union

import numpy as np


def generate_distance_based_edges(
    node_coords: np.ndarray, indices: Union[np.ndarray, List[int]], sections: List[int], nodes_per_section: List[int]
) -> np.ndarray:
    """Generates edge nodes based on distance from given node coordinates and indices.

    Parameters:
    ----------
    node_coords: np.ndarray
        A 2D numpy array of shape (num_nodes, 2) representing the coordinates of the nodes.
    indices: np.ndarray
        A 1D numpy array of shape (num_indices,) representing the indices of the nodes to consider.
    sections: List[int]
        A list of integers defining the boundaries of sections within the indices array.
    nodes_per_section: List[int]
        A list of integers specifying the maximum number of nodes to select from each section.

    Returns:
    np.ndarray: A 2D numpy array of shape (num_indices, num_selected_nodes) containing the sorted indices of
                the selected nodes based on distance.
    """

    # Calculate the relative positions of all nodes w.r.t the selected indices
    relative_positions = node_coords[indices, :, np.newaxis, :] - node_coords[indices, np.newaxis, :, :]

    # Calculate the Euclidean distances for these relative positions
    relative_distance = np.sqrt(np.sum(np.square(relative_positions), axis=-1, keepdims=True))

    # Sort indices based on the calculated distances
    sorted_indices = np.argsort(relative_distance.squeeze(axis=-1), axis=-1)[..., 1:]  # excluding the node itself

    # Select random nodes from the sorted indices
    sorted_indices = _random_select_nodes_by_sections(sorted_indices, sections, nodes_per_section)

    return sorted_indices


def _random_select_nodes_by_sections(
    indices: np.ndarray, sections: List[int], nodes_per_section: List[int]
) -> np.ndarray:
    """
    Randomly selects nodes from the given indices array based on predefined sections and selection counts.

    Parameters:
    ----------
    indices: np.ndarray
        A 3D numpy array of shape (sample_num, rows, cols) from which to select nodes.
    sections: List[int]
        A list of integers defining the boundaries of sections within the indices array.
    nodes_per_section: List[int]
        A list of integers specifying the maximum number of nodes to select from each section.

    Returns:
    ----------
    np.ndarray
        A 3D numpy array of selected values with shape (sample_num, rows, num_select_total).

    Example:
    ----------
    indices = np.random.randint(0, 1000, (10, 20, 1000))
    sections = [0, 20, 100, 200, 500, 1000]
    max_select_node = [20, 30, 30, 10, 10]
    selected_indices = _random_select_nodes(indices, sections, max_select_node)
    """
    num_samples, num_rows, num_cols = indices.shape

    # Pre-allocate selected_indices array
    total_selected_nodes = sum(nodes_per_section)

    selected_indices = np.zeros((num_samples, num_rows, total_selected_nodes), dtype=np.int32)

    for i in range(len(sections) - 1):
        start_idx = 0 if i == 0 else sum(nodes_per_section[:i])

        for b in range(num_samples):
            for r in range(num_rows):
                shuffle_indices = np.random.permutation(np.arange(sections[i], sections[i + 1]))[: nodes_per_section[i]]
                selected_indices[b, r, start_idx : start_idx + nodes_per_section[i]] = shuffle_indices  # noqa

    # Gather the selected indices from the original indices
    samples_indices = np.arange(num_samples)[:, None, None]

    row_indices = np.arange(num_rows)[None, :, None]

    return indices[samples_indices, row_indices, selected_indices]
