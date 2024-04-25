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

mp = {
    'MPCFormer':  ['ABY', 'SPDZ', 'ABY3', 'Falcon'],
    'resnet': ['ABY', 'SPDZ', 'ABY3', 'Falcon']
}

rename = {
    'MPCFormer':  'Bert',
    'resnet': 'ResNet'
}

ops = {
    'conv2d': 0,
    'softmax': 0,
    'gelu': 0,
    'relu': 0,
    'max_pool2d': 0,
    'mm': 0,
    'norm': 0,
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
    operations = set(key for data_dict in data_list for key in data_dict.keys())

    # 创建一个字典，用于存储每个操作的总时间
    total_times = {operation: [] for operation in operations}

    # 将每组数据中的每个操作的时间加到总时间中
    for data_dict in data_list:
        for operation, time in data_dict.items():
            total_times[operation].append(time)

    # 创建另一个 Figure 对象，用于绘制堆叠的柱状图
    fig, ax = plt.subplots()

    colors = [
        "#8ECFC9",
        "#FFBE7A",
        "#FA7F6F",
        "#82B0D2",
        "#BEB8DC",
        "#E7DAD2"
    ]




    
    # 遍历每个操作，绘制堆叠的柱状图
    bottom = np.zeros(len(name_list))
    for i, operation in enumerate(operations):
        ax.bar(wrap_labels(name_list), total_times[operation], label=operation, bottom=bottom, width=0.5, color=colors[i % len(colors)])
        bottom += np.array(total_times[operation])

    # 在 y 轴的刻度上绘制虚线
    for tick in ax.get_yticks():
        ax.axhline(y=tick, linestyle='dashed', color='gray', alpha=0.1)

    fig.legend(loc='upper center', ncol=3)
    
    # 格式化 y 轴标签为百分比
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # 保存图形到文件
    fig.savefig('2m4p.png')
    fig.savefig('2m4p.pdf')




if __name__=='__main__':
    data_list = []
    name_list = []
    for m, plist in mp.items():
        for p in plist:
            file = file_path + 'autograd_' + m + '-' + p + '.json'
            profiling_dicts = jsonLoader(file)
            profiling_comm = profiling_dicts['online communicationbits']
            #print(profiling_comm)
            
            profiling_ops = copy.deepcopy(ops)
            for it in profiling_comm:
                for op in profiling_ops:
                    if op in it[0]:
                        profiling_ops[op] += it[1]
            #print(profiling_ops)
            
            # change to percent
            summary = sum(profiling_ops.values())
            for k in profiling_ops:
                profiling_ops[k] = profiling_ops[k] / summary * 100
            profiling_ops['other_linear'] = profiling_ops['mm'] + profiling_ops['norm']
            profiling_ops.pop('mm', None)
            profiling_ops.pop('norm', None)
            print(profiling_ops)
            
            data_list.append(profiling_ops)

            name_list.append(rename[m] + '\n(' + p + ')')
    drawing_graph(data_list, name_list)       
    