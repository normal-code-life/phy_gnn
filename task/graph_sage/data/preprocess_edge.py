import numpy as np
#
# # 定义数据形状
data_shape = (2250, 6784, 6784)
#
# # 二进制文件名
input_filename = '/Users/mumin/Workplace/self/phy_gnn/pkg/data/lvData/processedData/TRAIN/node_neighbours_all_distance_TRAIN.npy'
#
# with open(input_filename, 'rb') as infile:
#     for i in range(0, data_shape[0], 3):
#         # 读取一行数据
#         row_data = np.fromfile(infile, dtype='float32', count=data_shape[1] * data_shape[2])
#         row_data = row_data.reshape((data_shape[1], data_shape[2]))
#
#         # 在这里进行数据处理（例如，这里简单的示例是将数据乘以2）
#         processed_row_data = row_data * 2
#
#         print("\n ddd", i, processed_row_data)





import torch
from torch.utils.data import IterableDataset, DataLoader

# 自定义 IterableDataset 类
import torch
from torch.utils.data import IterableDataset, DataLoader


# 自定义 IterableDataset 类
class CustomIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        # 打开文件并逐行读取数据
        with open(input_filename, 'rb') as infile:
            for i in range(0, data_shape[0]):
                # 读取一行数据
                row_data = np.fromfile(infile, dtype='int32', count=data_shape[1] * data_shape[2])
                row_data = row_data.reshape((data_shape[1], data_shape[2]))

                yield row_data

# 创建 IterableDataset 实例
dataset = CustomIterableDataset(input_filename)

# 创建 DataLoader
batch_size = 20
dataloader = DataLoader(dataset, batch_size=batch_size)

# 使用 DataLoader 迭代数据
i = 0
for batch in dataloader:
    print("Batch shape:", batch.shape)  # batch 是一个形状为 (batch_size, seq_len) 的张量
    # print("Batch data:", batch)
    i += batch.shape[0]
    print(i)
