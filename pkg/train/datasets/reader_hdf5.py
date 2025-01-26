"""HDF5 File Reader.

The current file creates an iterator that reads and merges data from multiple HDF5 datasets based on a given pattern.
Functions below are useful for handling and processing data spread across multiple HDF5 files,
and it supports both standard examples and sequence examples.

Note: the main logic originates from the TFRecord repository.
"""
import functools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Set, Tuple, Union

import h5py
import numpy as np

from pkg.utils.logs import init_logger

logger = init_logger("HDF5_LOADER")


def multi_hdf5_loader(
    data_pattern: str,
    splits: Set[str],
    description: Union[List[str], Dict[str, str], None] = None,
    sequence_description: Union[List[str], Dict[str, str], None] = None,
    infinite: bool = False,
) -> Iterable[Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, List[np.ndarray]]],]]:
    """Create an iterator by reading and merging multiple HDF5 datasets.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    Returns:
    --------
    it: iterator
        A repeating iterator that generates batches of data.
    """
    loaders = [
        functools.partial(
            single_hdf5_loader,
            data_path=data_pattern.format(split),
            description=description,
            sequence_description=sequence_description,
        )
        for split in splits
    ]

    return sample_iterators(loaders, infinite)


def single_hdf5_loader(
    data_path: str,
    description: Union[List[str], Dict[str, str], None] = None,
    sequence_description: Union[List[str], Dict[str, str], None] = None,
) -> Iterable[Dict[str, np.ndarray]]:
    """Create an iterator over the hdf5 file contained within the dataset.

    Decodes raw bytes of both the context and features (contained within the
    dataset) into its respective format.

    Params:
    -------
    data_path: str
        hdf5 file path.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    Yields:
    -------
    A dict of features for an individual record.
    """
    with h5py.File(data_path, "r") as f:
        for sample_name in f.keys():
            context_example = {}
            feature_example = {}

            group = f[sample_name]

            for key in group.keys():
                if key in description:
                    context_example[key] = group[key][:]

                if key in sequence_description:
                    feature_example[key] = group[key][:]

            yield context_example, feature_example


def cycle(iterator_fn: Callable) -> Iterable[Any]:
    """Create a repeating iterator from an iterator generator."""
    while True:
        for element in iterator_fn():
            yield element


def sample_iterators(iterators: List[Iterator], infinite: bool = False) -> Iterable[Any]:
    """Retrieve info generated from the iterator(s) according to their sampling ratios.

    Params:
    -------
    iterators: list of iterators
        All iterators (one for each file).

    ratios: list of int
        The ratios with which to sample each iterator.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """
    if infinite:
        iterators = [cycle(iterator) for iterator in iterators]
    else:
        iterators = [iterator() for iterator in iterators]

    while iterators:
        choice = np.random.choice(len(iterators))
        try:
            yield next(iterators[choice])
        except StopIteration:
            if iterators:
                del iterators[choice]


def shuffle_iterator(iterator: Iterator, queue_size: int) -> Iterable[Any]:
    """Shuffle elements contained in an iterator.

    Params:
    -------
    iterator: iterator
        The iterator.

    queue_size: int
        Length of buffer. Determines how many records are queued to
        sample from.

    Yields:
    -------
    item: Any
        Decoded bytes of the features into its respective data type (for
        an individual record) from an iterator.
    """
    buffer = []
    try:
        for _ in range(queue_size):
            buffer.append(next(iterator))
    except StopIteration:
        logger.warn("Number of elements in the iterator is less than the queue size (N={queue_size}).")
    while buffer:
        index = np.random.randint(len(buffer))
        try:
            item = buffer[index]
            buffer[index] = next(iterator)
            yield item
        except StopIteration:
            yield buffer.pop(index)
