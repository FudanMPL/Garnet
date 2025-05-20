import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json
import copy
import numpy as np
import textwrap

file_path = './'
file_type = '.json'
fig_type = '.png'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13  # 设置字体大小
# plt.rcParams['font.weight'] = 'bold'  # 设置字体加粗

mp = {
    'resnet': ['ABY', 'SPDZ', 'ABY3', 'Falcon'],
    'MPCFormer':  ['ABY', 'SPDZ', 'ABY3', 'Falcon'],
}

m_rename = {
    'resnet': 'ResNet-50',
    'MPCFormer':  r'BERT$_{\mathrm{BASE}}$',
}

p_rename = {
    'ABY': 'ABY',
    'SPDZ': 'SPDZ-2k',
    'ABY3': 'ABY3',
    'Falcon': 'Falcon'
}

ops = {
    'conv2d': 'Conv2d',
    'softmax': 'Softmax',
    'gelu': 'Gelu',
    'relu': 'Relu',
    'max_pool2d': 'MaxPool',
    'mm': 'MatMul',
    'batch_norm': 'BatchNorm',
    'layer_norm': 'LayerNorm',
}

colors = [
    "#8ECFC9",
    "#FFBE7A",
    "#FA7F6F",
    "#82B0D2",
    "#F6CAE5",
    "#E7DAD2",
    "#CFEBF1",
    "#A2AAD1"
]

color_dict = {
    'Conv2d': colors[0],
    'Softmax': colors[3],
    'Gelu': colors[5],
    'Relu': colors[7],
    'MaxPool': colors[6],
    'MatMul': colors[1],
    'BatchNorm': colors[2],
    'LayerNorm': colors[4],
}

hatch_dict = {
    'Conv2d': '/',
    'Softmax': '\\',
    'Gelu': '-',
    'Relu': '.',
    'MaxPool': '|',
    'MatMul': 'x',
    'BatchNorm': 'o',
    'LayerNorm': 'O',
}

def jsonLoader(file_name):
    with open(file_name, 'r') as json_file:
        json_data = json.load(json_file)
        return json_data
    raise 'FILE ACCESS ERROR'




def wrap_labels(labels, width=8):
    return [textwrap.fill(label, width) for label in labels]

def percentage_formatter(x, pos):
    return f'{x:.0f}%'

def drawing_graph(data_list, name_list):
    # 提取所有可能的操作名称
    operations = ['Relu', 'Softmax', 'LayerNorm', 'MaxPool', 'BatchNorm', 'MatMul','Gelu', 'Conv2d']

    # 创建一个字典，用于存储每个操作的总时间
    total_times = {operation: [] for operation in operations}

    # 将每组数据中的每个操作的时间加到总时间中
    for data_dict in data_list:
        for operation, time in data_dict.items():
            total_times[operation].append(time)

    # 创建另一个 Figure 对象，用于绘制堆叠的柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 遍历每个操作，绘制堆叠的柱状图
    bottom = np.zeros(len(name_list))
    for i, operation in enumerate(operations):
        ax.bar(name_list, total_times[operation], label=operation, bottom=bottom, width=0.6, color=color_dict.get(operation, 'gray'), hatch=hatch_dict.get(operation, '/'), alpha=1)
        bottom += np.array(total_times[operation])

    # 在 y 轴的刻度上绘制虚线
    for tick in ax.get_yticks():
        ax.axhline(y=tick, linestyle='dashed', color='gray', alpha=0.15)

    fig.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.01), prop={'size': 16}, handlelength=2)
    
    # 格式化 y 轴标签为百分比
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # 去掉框线
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 调整子图的边距
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.08, right=1)
    
    # 保存图形到文件
    fig.savefig('protocol_profiling.png')
    fig.savefig('protocol_profiling.pdf')




if __name__=='__main__':
    data_list = []
    name_list = []
    for m, plist in mp.items():
        for p in plist:
            file = file_path + 'autograd_' + m + '-' + p + '.json'
            profiling_dicts = jsonLoader(file)
            profiling_comm = profiling_dicts['online communicationbits']
            #print(profiling_comm)
            
            profiling_ops = {}
            for name, rename in ops.items():
                profiling_ops[rename] = 0
            for it in profiling_comm:
                for name, rename in ops.items():
                    if name in it[0]:
                        profiling_ops[rename] += it[1]
            # print(profiling_ops)
            
            # change to percent
            summary = sum(profiling_ops.values())
            for k in profiling_ops:
                profiling_ops[k] = profiling_ops[k] / summary * 100
            print(profiling_ops)
            
            data_list.append(profiling_ops)

            name_list.append(m_rename[m] + '\n(' + p_rename[p] + ')')
    drawing_graph(data_list, name_list)       
    