import threading
import time

import numpy as np
from numba.typed import List

from pkg.data.utils.edge_generation import generate_distance_based_edges_nb, generate_distance_based_edges_ny
from pkg.utils.monitor import monitor_cpu_usage


def test_generate_distance_based_edges():
    batch_size = 1

    for i in range(0, batch, batch_size):
        end = min(i + batch_size, num_nodes)

        indices = [idx for idx in range(i, end)]

        sorted_indices_ny = generate_distance_based_edges_ny(node_coords, indices, sections, nodes_per_section)

        print("test_generate_distance_based_edges: ", i, sorted_indices_ny.shape)

    return sorted_indices_ny


def test_generate_distance_based_edges_v2():
    sections_nb = List()
    [sections_nb.append(x) for x in sections]

    nodes_per_section_nb = List()
    [nodes_per_section_nb.append(x) for x in nodes_per_section]

    for i in range(batch):
        sorted_indices_nb = generate_distance_based_edges_nb(node_coords[i], sections_nb, nodes_per_section_nb)

        print("test_generate_distance_based_edges_v2: ", i, sorted_indices_nb.shape, type(sorted_indices_nb[0][0]))

    return sorted_indices_nb


if __name__ == "__main__":
    start_time = time.time()

    num_nodes = 250
    batch = 10
    sections = [0, 20, 50, 100, 200]
    nodes_per_section = [20, 10, 30, 30]

    node_coords = np.random.rand(batch, num_nodes, 3)

    monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(10,))
    monitor_thread.start()

    # res = test_generate_distance_based_edges()

    res1 = test_generate_distance_based_edges_v2()

    print(f"{time.time() - start_time}s")

    # print(f"res: {res}")
    # print(f"res1: {res1}")
    # print(f"{(sorted(res[0][0][: 20]) == sorted(res1[0][: 20]))}")  # noqa

    # Wait for the monitor to finish
    monitor_thread.join()
