import torch


def segment_sum(data: torch.tensor, segment_ids: torch.tensor, num_segments: int) -> torch.tensor:
    """Computes the sum of each segment within the given data.

    It assumes that the data has already been grouped into segments,
    and each data element has a corresponding segment ID.
    The function takes three parameters as input:

    :param data: A tensor containing the actual data.
    :param segment_ids: A tensor with same length as 'data', indicating the segment to which each data element belongs.
    :param num_segments: The total number of segments.
    :return: the function returns the tensor 'result' containing the sum of each segment.
    """
    result = torch.zeros(num_segments, dtype=data.dtype, device=data.device)
    result.index_add_(0, segment_ids, data)
    return result
