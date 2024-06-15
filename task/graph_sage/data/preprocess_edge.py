import numpy as np

# 定义数据形状
data_shape = (2250, 6784, 3)

# 二进制文件名
input_filename = '/Users/mumin/Workplace/self/phy_gnn/pkg/data/lvData/processedData/TRAIN/node_neighbours_distance_train.npy'

with open(input_filename, 'rb') as infile:
    for i in range(0, data_shape[0], 3):
        # 读取一行数据
        row_data = np.fromfile(infile, dtype='float32', count=data_shape[1] * data_shape[2])
        row_data = row_data.reshape((data_shape[1], data_shape[2]))

        # 在这里进行数据处理（例如，这里简单的示例是将数据乘以2）
        processed_row_data = row_data * 2

        print("\n ddd", i, processed_row_data)