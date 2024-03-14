import torch
from pkg.tf_utils.method import segment_sum


def test_segment_sum():
    # Example data
    data = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30]
         ]
    )
    segment_ids = torch.tensor([0, 0, 1, 1, 2, 2])  # Indicates the segment each element belongs to
    num_segments = 3  # Total number of segments

    # Call segment_sum function
    segment_sums = segment_sum(data, segment_ids, num_segments)
    print("Sum of each segment:", segment_sums)



if __name__ == "__main__":
    test_segment_sum()
