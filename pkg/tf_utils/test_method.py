import torch
from pkg.tf_utils.method import segment_sum


def test_segment_sum():
    # Example data
    data = torch.tensor([1, 2, 3, 4, 5, 6])
    segment_ids = torch.tensor([0, 0, 1, 1, 2, 2])  # Indicates the segment each element belongs to
    num_segments = 3  # Total number of segments

    # Call segment_sum function
    segment_sums = segment_sum(data, segment_ids, num_segments)
    print("Sum of each segment:", segment_sums)
    assert torch.all(torch.eq(segment_sums, torch.tensor([3, 7, 11]))), "test_segment_sum method failed"


if __name__ == "__main__":
    test_segment_sum()
