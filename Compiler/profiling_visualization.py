
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

name_dict = {"online communicationbits":"online comm",
             "offline communicationbits":"offline comm",
             "onlineround":"online round",
             "offlineround":"offline round"}
def parse(profiling_res):
    res = defaultdict(lambda: -1)
    for key, value in profiling_res.items():
        for req, num in value.items():
            if  "online" in req[0] or  "offline" in req[0]:
                if res[req[0]+req[1]] == -1:
                    res[req[0]+req[1]] = [(key, num)] 
                else:
                    res[req[0]+req[1]].append((key,num)) 
    return res


font_legend = {'family': 'Times New Roman','weight': 'normal', 'size': 18}
font_label = {'family': 'Times New Roman','weight': 'normal', 'size': 16}
colors = ['red','blue','green','orange','purple']
dir = "Compiler/profiling_res/"

def plot_histogram(data, name):
    plt.xlabel('Component', font_label)
    plt.ticklabel_format(axis='x', style='sci',scilimits=(0, 0))
    for key, value in data.items():
        plt.clf()
        plt.yscale('log')
        plt.ylabel(name_dict[key], font_label)
        name_list = []
        num_list = []
        for x in value:
            name_list.append(x[0])
            num_list.append(x[1])
        plt.bar(range(len(num_list)), num_list, tick_label = name_list)
        plt.xlabel('Component', font_label)
        plt.tight_layout()
        plt.savefig(dir+name+" "+key+".pdf")

def plot_stackedcolumn(data, name):
    plt.clf()
    plt.xlabel('Phase', font_label)
    plt.ylabel('Cost', font_label)
    # plt.ticklabel_format(axis='x', style='sci',scilimits=(0, 0))
    store = defaultdict(lambda: -1)
    x_label = []
    for key, value in data.items():
        x_label.append(name_dict[key])
        sum = 0
        for x in value:
            sum += x[1]
        if sum == 0:
            store[x[0]].append(1/len(data.keys()))
            continue
        for x in value:
            if store[x[0]] == -1:
                store[x[0]] = [x[1]/sum]
                continue
            store[x[0]].append(x[1]/sum)
    i = 0
    bottom = []
    for key, value in store.items():
        if i == 0:
            plt.bar(x_label, value,width=0.4,label=key,color=colors[i],edgecolor='grey',zorder=5)
        else:
            plt.bar(x_label, value,width=0.4, bottom=bottom,label=key,color=colors[i],edgecolor='grey',zorder=5)
        bottom = np.sum([bottom,value], axis=0)
        i+=1
    plt.ylim(0,1.01)
    plt.yticks(np.arange(0,1.2,0.2),[f'{i}%' for i in range(0,120,20)])
    plt.grid(axis='y',alpha=0.5,ls='--')
    plt.grid(axis='x',alpha=0.5,ls='--')
    plt.legend(frameon=False,bbox_to_anchor=(1.01,1))
    plt.tight_layout()
    plt.savefig(dir+name+"-stack.pdf")


file_path = './Programs/Profiling-data/'
file_type = '.json'

def plot_cost(profiling_res, name, protocol):
    data = parse(profiling_res)
    file_id = name + '-' + protocol
    print('using <python ./Scripts/profiling-v.py '+file_id+'> to visualize the result.')
    file_name = file_path + file_id + file_type
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=2)
    # plot_histogram(data, name)
    # plot_stackedcolumn(data, name)
