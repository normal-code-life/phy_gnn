"""Tests for edge generation functions.

This module contains test functions to compare and benchmark different implementations
of distance-based edge generation algorithms. It tests both numpy and numba implementations
and measures their performance and CPU usage.
"""

import threading
import time
from typing import List as TypedList

import numpy as np
from numba.typed import List

from pkg.data_utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.utils.monitor import monitor_cpu_usage


def test_generate_distance_based_edges(
    node_coords: np.ndarray,
    batch: int,
    num_nodes: int,
    sections: TypedList[int],
    nodes_per_section: TypedList[int],
    batch_size: int = 1,
) -> np.ndarray:
    """Test the numpy implementation of distance-based edge generation.

    Args:
        node_coords: Node coordinates array of shape (batch, num_nodes, 3)
        batch: Number of batches to process
        num_nodes: Number of nodes per batch
        sections: List of section boundaries
        nodes_per_section: List of nodes per section
        batch_size: Size of each processing batch

    Returns:
        np.ndarray: Generated edge indices
    """
    for i in range(0, batch, batch_size):
        end = min(i + batch_size, num_nodes)
        indices = [idx for idx in range(i, end)]
        sorted_indices_ny = generate_distance_based_edges_ny(node_coords, indices, sections, nodes_per_section)
        print("test_generate_distance_based_edges: ", i, sorted_indices_ny.shape)

    return sorted_indices_ny


def test_generate_distance_based_edges_numba(
    node_coords: np.ndarray,
    batch: int,
    sections: TypedList[int],
    nodes_per_section: TypedList[int],
) -> np.ndarray:
    """Test the numba implementation of distance-based edge generation.

    Args:
        node_coords: Node coordinates array of shape (batch, num_nodes, 3)
        batch: Number of batches to process
        sections: List of section boundaries
        nodes_per_section: List of nodes per section

    Returns:
        np.ndarray: Generated edge indices
    """
    sections_nb = List()
    [sections_nb.append(x) for x in sections]

    nodes_per_section_nb = List()
    [nodes_per_section_nb.append(x) for x in nodes_per_section]

    for i in range(batch):
        sorted_indices_nb = generate_distance_based_edges_nb(node_coords[i], sections_nb, nodes_per_section_nb)
        print(
            "test_generate_distance_based_edges_numba: ",
            i,
            sorted_indices_nb.shape,
            type(sorted_indices_nb[0][0]),
        )

    return sorted_indices_nb


if __name__ == "__main__":
    # Test configuration
    start_time = time.time()
    num_nodes = 250
    batch = 10
    sections = [0, 20, 50, 100, 200]
    nodes_per_section = [20, 10, 30, 30]

    # Generate random test data
    node_coords = np.random.rand(batch, num_nodes, 3)

    # Start CPU monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(10,))
    monitor_thread.start()

    # Run tests
    res = test_generate_distance_based_edges(node_coords, batch, num_nodes, sections, nodes_per_section)

    # Uncomment to test numba implementation
    # res1 = test_generate_distance_based_edges_numba(node_coords, batch, sections, nodes_per_section)

    print(f"Total execution time: {time.time() - start_time}s")

    # Uncomment to compare results
    # print(f"Numpy result: {res}")
    # print(f"Numba result: {res1}")
    # print(f"First 20 elements match: {(sorted(res[0][0][: 20]) == sorted(res1[0][: 20]))}")  # noqa

    # Wait for the CPU monitor to finish
    monitor_thread.join()
