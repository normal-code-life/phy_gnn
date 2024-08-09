"""
This module is primarily used to generate edge nodes for a given node.
It is designed under the assumption that any node has the potential to be connected,
And this approach is based on location information, randomly selecting nearby points.

The difference between generate_distance_based_edges_ny and generate_distance_based_edges_nb is that
generate_distance_based_edges_ny is built using NumPy, while generate_distance_based_edges_nb is built using JIT.
The `nb` version is more suitable for large-scale datasets which will leverge the multi-cpu

potential issue: https://github.com/pytorch/pytorch/issues/78490
"""

import warnings
from typing import List, Union

import numpy as np
from numba import njit, prange
from numba.core.errors import NumbaTypeSafetyWarning
from numba.typed import List as Numba_List

from pkg.math import numba as nb

warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)


# fmt: off

def generate_distance_based_edges_ny(
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
    return _random_select_nodes_by_sections_ny(sorted_indices, sections, nodes_per_section)


def _random_select_nodes_by_sections_ny(
    indices: np.ndarray, sections: Numba_List[int], nodes_per_section: Numba_List[int]
) -> np.ndarray:
    """
    Randomly selects nodes from the given indices array based on predefined sections and selection counts.

    Parameters:
    ----------
    indices: np.ndarray
        A 3D numpy array of shape (sample_num, rows, cols) from which to select nodes.
    sections: Bumba_List[int]
        A list of integers defining the boundaries of sections within the indices array.
    nodes_per_section: Bumba_List[int]
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
                selected_indices[b, r, start_idx: start_idx + nodes_per_section[i]] = np.random.permutation(
                    indices[b, r, sections[i]: sections[i + 1]]
                )[: nodes_per_section[i]]

    return selected_indices


@njit(parallel=True)
def generate_distance_based_edges_nb(
    node_coords: np.ndarray, sections_nb: Numba_List[int], nodes_per_section_nb: Numba_List[int]
) -> np.ndarray:
    """Generates edge nodes based on distance from given node coordinates and indices.

    Parameters:
    ----------
    node_coords: np.ndarray
        A 2D numpy array of shape (num_nodes, 2) representing the coordinates of the nodes.
    indices: np.ndarray
        A 1D numpy array of shape (num_indices,) representing the indices of the nodes to consider.
    sections: Bumba_List[int]
        A list of integers defining the boundaries of sections within the indices array.
    nodes_per_section: Bumba_List[int]
        A list of integers specifying the maximum number of nodes to select from each section.

    Returns:
    np.ndarray: A 2D numpy array of shape (num_indices, num_selected_nodes) containing the sorted indices of
                the selected nodes based on distance.
    """
    num_nodes = node_coords.shape[0]

    # Calculate the relative positions of all nodes w.r.t the selected indices
    relative_distance = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    sorted_indices = np.zeros((num_nodes, num_nodes - 1), dtype=np.int64)

    for i in prange(num_nodes):
        for j in range(num_nodes):
            relative_distance[i, j] = np.sqrt(np.sum(np.square(node_coords[i] - node_coords[j])))

    # Sort indices based on the calculated distances
    for i in prange(num_nodes):
        sorted_indices[i] = nb.argsort(relative_distance[i])[..., 1:]  # excluding the node itself

    # Select random nodes from the sorted indices
    return _random_select_nodes_by_sections_nb(sorted_indices, sections_nb, nodes_per_section_nb)


@njit(parallel=True)
def _random_select_nodes_by_sections_nb(
    indices: np.ndarray, sections_nb: List[int], nodes_per_section_nb: List[int]
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
    num_nodes = indices.shape[0]
    indices = indices.astype(np.int64)
    # Pre-allocate selected_indices array
    total_selected_nodes = sum(nodes_per_section_nb)
    selected_indices = np.zeros((num_nodes, total_selected_nodes), dtype=np.int64)

    for i in prange(len(sections_nb) - 1):
        start_idx = 0 if i == 0 else sum(nodes_per_section_nb[:i])

        for n in range(num_nodes):
            selected_indices[n, start_idx: start_idx + nodes_per_section_nb[i]] = np.random.choice(
                indices[n, sections_nb[i]: sections_nb[i + 1]], nodes_per_section_nb[i], False
            )

    return selected_indices

# fmt: on
