import torch
from pkg.tf_utils.method import segment_sum


def test_segment_sum():
    # Example data
    # data = torch.tensor(
    #     [
    #         [1, 2, 3, 4, 5],
    #         [6, 7, 8, 9, 10],
    #         [11, 12, 13, 14, 15],
    #         [16, 17, 18, 19, 20],
    #         [21, 22, 23, 24, 25],
    #         [26, 27, 28, 29, 30]
    #      ]
    # )
    # segment_ids = torch.tensor([0, 0, 1, 1, 2, 2])  # Indicates the segment each element belongs to
    # num_segments = 3  # Total number of segments
    #
    # # Call segment_sum function
    # segment_sums = segment_sum(data, segment_ids, num_segments)
    # print("Sum of each segment:", segment_sums)
    # assert torch.all(torch.eq(segment_sums, torch.tensor([3, 7, 11]))), "test_segment_sum method failed"

    x = torch.ones(5, 3)
    print(x)
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    index = torch.tensor([0, 4, 2])
    print(x.index_add_(0, index, t))

if __name__ == "__main__":
    test_segment_sum()
