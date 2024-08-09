import time

import numpy as np

from pkg.math.numba.sort import argsort as nb_argsort


def accuracy_test():
    data = np.array([30, 10, 50, 20, 40])
    sorted_indices_numba = nb_argsort(data.copy())
    sorted_indices_numpy = np.argsort(data)

    assert (sorted_indices_numba == sorted_indices_numpy).all()  # noqa

    print(f"original_data: {data}")
    print(f"rank index by numba: {sorted_indices_numba}")
    print(f"rank index by numpy: {sorted_indices_numpy}")
    print(f"data based on rank index: {[data[i] for i in sorted_indices_numba]}")


def efficiency_test():
    for sample in [100, 1000, 10000, 1000000, 10000000, 1000000000]:
        start_time = time.time()

        array = np.random.random(sample)

        nb_argsort(array)

        print(f"{sample} samples nb_argsort takes {time.time() - start_time:.4f}s")

        start_time = time.time()

        array = np.random.random(sample)

        np.argsort(array)

        print(f"{sample} samples np_argsort takes {time.time() - start_time:.4f}s")


if __name__ == "__main__":
    # accuracy_test()
    """accuracy_test results.
    original_data: [30 10 50 20 40]
    rank index by numba: [1 3 0 4 2]
    rank index by numpy: [1 3 0 4 2]
    data based on rank index: [10, 20, 30, 40, 50]
    """

    efficiency_test()
    """efficiency_test result
    100 samples nb_argsort takes 0.2300s
    100 samples np_argsort takes 0.0000s
    1000 samples nb_argsort takes 0.0001s
    1000 samples np_argsort takes 0.0000s
    10000 samples nb_argsort takes 0.0007s
    10000 samples np_argsort takes 0.0006s
    1000000 samples nb_argsort takes 0.0940s
    1000000 samples np_argsort takes 0.0882s
    10000000 samples nb_argsort takes 1.0847s
    10000000 samples np_argsort takes 1.1604s
    1000000000 samples nb_argsort takes 136.6169s
    1000000000 samples np_argsort takes 201.8790s
    """
