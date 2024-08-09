import numpy as np
from numba import njit

# To fully leverage the capabilities of multi-core CPUs, we use Numba to process certain large-scale datasets.


@njit
def quick_sort(arr: np.ndarray, indices: np.ndarray, low: int, high: int) -> None:
    """Perform in-place quicksort on the array and indices.

    Parameters:
    ----------
    arr : np.ndarray
        The array to be sorted.
    indices : np.ndarray
        An array of indices that correspond to the positions of the elements in the original array.
    low : int
        The starting index for the sorting.
    high : int
        The ending index for the sorting.
    """
    if low < high:
        pi = partition(arr, indices, low, high)

        quick_sort(arr, indices, low, pi - 1)

        quick_sort(arr, indices, pi + 1, high)


@njit
def partition(arr: np.ndarray, indices: np.ndarray, low: int, high: int) -> int:
    """Partition the array around a pivot element.

    Parameters:
    ----------
    arr : np.ndarray
        The array to be partitioned.
    indices : np.ndarray
        An array of indices that correspond to the positions of the elements in the original array.
    low : int
        The starting index for the partitioning.
    high : int
        The ending index for the partitioning.

    Returns:
    -------
    int
        The index of the pivot element after partitioning.
    """
    pivot = arr[high]

    i = low - 1

    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            indices[i], indices[j] = indices[j], indices[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    indices[i + 1], indices[high] = indices[high], indices[i + 1]

    return i + 1


@njit
def argsort(array: np.ndarray) -> np.ndarray:
    """Return the indices that would sort an array using quicksort.

    Parameters:
    ----------
    array : np.ndarray
        The array to be sorted.

    Returns:
    -------
    np.ndarray
        An array of indices that sorts the input array.
    """
    indices = np.arange(len(array), dtype=np.int32)

    quick_sort(array, indices, 0, len(array) - 1)

    return indices
