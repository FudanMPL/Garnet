import sys
import matplotlib.pyplot as plt
import json


file_path = './Programs/Profiling-data/'
file_type = '.json'
fig_type = '.png'


def jsonLoader(json_name):
    with open(file_name, 'r') as json_file:
        json_data = json.load(json_file)
        return json_data
    raise 'FILE ACCESS ERROR'

def visualizing(single_dict, fig_name):
    # 提取"online communicationbits"的值
    online_communicationbits_values = [item[1] for item in single_dict if item[1] != 0]

    # 提取"online communicationbits"的标签
    online_communicationbits_labels = [item[0] for item in single_dict if item[1] != 0]

    # 按值从大到小排序
    sorted_indices = sorted(range(len(online_communicationbits_values)), key=lambda k: online_communicationbits_values[k], reverse=True)
    values = [online_communicationbits_values[i] for i in sorted_indices]
    labels = [online_communicationbits_labels[i] for i in sorted_indices]
    print(values)
    print(labels)

    # 创建圆盘比例可视化图表
    plt.figure(figsize=(8, 8))
    colors = plt.cm.Paired(range(len(values)))
    plt.pie(values, colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'), autopct='%1.1f%%', startangle=90, pctdistance=0.85)

    # 添加颜色+标签的说明栏
    handles = [plt.Rectangle((0,0),1,1, color=colors[i], ec="k") for i in range(len(values))]

    # 将说明栏放在右侧
    # plt.legend(handles, labels, bbox_to_anchor=(0, 0), loc='upper left', borderaxespad=0)
    legend = plt.legend(handles, labels, bbox_to_anchor=(0.5, 0.1), loc='upper center', borderaxespad=0., bbox_transform=plt.gcf().transFigure, ncol=3)

    # 添加标题
    plt.title('Online Communication Bits')

    # 保存图表
    plt.savefig(fig_name)


if __name__=='__main__':
    script_name = sys.argv[0]
    arguments = sys.argv[1:]
    
    level = 2
    if len(arguments)>1:
        level = int(arguments[1])
    
    file_name = file_path + arguments[0] + file_type
    fig_name = file_path + arguments[0] + fig_type

    profiling_dict = jsonLoader(file_name)
    onlineCOMM_dict = profiling_dict['online communicationbits']
    
    result_dict = {}

    for item in onlineCOMM_dict:
        keys = item[0].split('-')
        
        key = ''
        for i in range(0, min(len(keys), level)):
            key += keys[i] + '-'
        
        if key in result_dict:
            result_dict[key] += item[1]
        else:
            result_dict[key] = item[1]
    result_list = [[key, value] for key, value in result_dict.items()]
    print(result_list)
    
    visualizing(result_list, fig_name)