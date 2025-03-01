import torch


def segment_sum(data: torch.tensor, segment_ids: torch.tensor, num_segments: int) -> torch.tensor:
    """Computes the sum along segments of a tensor.

    Sums values in 'data' tensor according to segment IDs, producing a tensor with one sum per segment.
    Only supports 2D input tensors.

    Args:
        data: Input tensor of shape (N, D) containing values to sum
        segment_ids: 1D tensor of shape (N,) with segment ID for each row in data
        num_segments: Total number of segments to compute sums for

    Returns:
        Tensor of shape (num_segments, D) containing segment sums

    Note:
        segment_ids must be in range [0, num_segments)
        segment_ids length must match first dimension of data
    """
    result = torch.zeros((num_segments, data.shape[-1]), dtype=data.dtype, device=data.device)
    result.index_add_(0, segment_ids, data)
    return result
