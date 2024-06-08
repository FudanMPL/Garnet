import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt('data')

# 提取每一列的数据
col1 = data[:, 0]
col2 = data[:, 1]
col3 = data[:, 2]

# max-0归一化函数
def max_zero_normalize(column):
    return column / column[0]

# 对每一列数据进行max-0归一化
col1_scaled = max_zero_normalize(col1) * 100
col2_scaled = max_zero_normalize(col2) * 100
col3_scaled = max_zero_normalize(col3) * 100

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制第一列数据
plt.plot(col1_scaled, label='round')

# 绘制第二列数据
plt.plot(col2_scaled, label='comm_bits')

# 绘制第三列数据
plt.plot(col3_scaled, label='predicted_lantency')

# 添加标题和标签
plt.title('MPL DNN Model Graph Optimizing')
plt.xlabel('searching graphs')
plt.ylabel('opt % of origin graph')

# 添加图例
plt.legend()

# 保存图形为PNG文件
plt.savefig('MPLTS.png')