"""Used to get the structured record from test raw data.
"""
import numpy as np
import copy
from NFGen.profiler import SubPoly
import pickle
import matplotlib.pyplot as plt


def fetch_result(file_path):
    """fetch time record from param - file_path.
    """
    time_list = []
    with open(file_path) as f:
        line = f.readline()
        while line:
            strs = line.split(" ")
            if (strs[0] == 'Stopped'):
                time_list.append(float(strs[4][:-1]))
            line = f.readline()
    time_list.insert(0, 0)
    time_list = np.diff(time_list)

    return time_list


def construct_profiler(time_list, building_blocks, repeat_times):
    """Construct the building blocks' profiling information.
    """
    bb_profiler = {key: [] for key in building_blocks}
    mapper = {i: building_blocks[i] for i in range(len(building_blocks))}

    for i in range(len(time_list)):
        which = mapper[i % len(building_blocks)]
        bb_profiler[which].append(time_list[i])

    # trans, as bits_compose contains the time of bits_decompose.
    bb_profiler['bit_compose'] = [
        bb_profiler['bit_compose'][i] - bb_profiler['bit_decompose'][i]
        for i in range(len(bb_profiler['bit_compose']))
    ]

    # trans - divide by the repeat times.
    for key in building_blocks:
        for i in range(len(bb_profiler[key])):
            bb_profiler[key][i] /= repeat_times[i]

    return bb_profiler


def separate_train_test(profiler_dict, building_blocks, ratio=0.1):
    """Separet teh train and test dataset.

    In the train and test sets, x is the data length, y is the time record for each building blocks.
    """
    data_len = len(profiler_dict['n_list'])
    test = int(data_len*ratio) if int(data_len*ratio) > 5 else 5
    data_proto = {key:[] for key in building_blocks}

    train_set = {'x':[], 'data':copy.deepcopy(data_proto)}
    test_set = {'x':[], 'data':copy.deepcopy(data_proto)}

    x = np.array(profiler_dict['n_list'])
    data = profiler_dict['data']
    indices = [i for i in range(data_len)]
    np.random.shuffle(indices)
    x = x[indices]

    train_set['x'] = x[:-test]
    test_set['x'] = x[-test:]

    for key in data.keys():
        darr = np.array(data[key])[indices]
        train_set['data'][key] = darr[:-test]
        test_set['data'][key] = darr[-test:]
    
    return train_set, test_set



def process_iterative(time_list, repeat_times):
    """Extract the recording time for ITERATIVE-Based functions, with division of repeat-times.
    """
    time_iterative = {  # the sequence is important - corresponding with the test.py.
        'sqrt_bits': [],
        'divide_bits': [],
        'sqrt_comp': [],
        'divide_comp': []
    }

    mapper = {i: list(time_iterative.keys())[i] for i in range(4)}
    for i in range(len(time_list)):
        time_iterative[mapper[i % 4]].append(time_list[i])

    for key in list(time_iterative.keys()):
        for i in range(len(time_iterative[key])):
            time_iterative[key][i] /= repeat_times[i]

    return time_iterative


def profiler_analysis(k_list, m_list, X, y, y_pred, save_path):
    """Care that the first dimension of X is m and the second dimension is k.
    """
    figure = plt.figure(figsize=(12, 6))
    figure.subplots_adjust(hspace=0.3, wspace=0.15)
    # plt.tick_params(labelsize=20)

    ax = figure.add_subplot(1, 2, 1)
    i = 0
    for k in k_list:
        index = [X[:, 1] == k]
        x_k = X[index][:, 0]
        y_k = y[index]
        ax.scatter(x_k, y_k, edgecolors=plt.cm.tab10(i), alpha=0.6, marker='^',facecolors='none', s=80)
        y_k = y_pred[index]
        ax.plot(x_k, y_k, label="k = %d"%k, color=plt.cm.tab10(i), linewidth=2)
        
        i += 1
    ax.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.legend(prop={'size':18})
    ax.set_xlabel("$m$", fontsize=20)
    ax.set_ylabel("Time(s)", fontsize=20)
    ax.set_title("M dimension", fontsize=20)


    ax = figure.add_subplot(1, 2, 2)
    i = 0
    for m in m_list:
        index = [X[:, 0] == m]
        x_k = X[index][:, 1]
        y_k = y[index]
        y_k = y_pred[index]
        ax.plot(x_k, y_k, label="m = %d"%m, color=plt.cm.tab10(i), linewidth=2)
        ax.scatter(x_k, y_k, edgecolors=plt.cm.tab10(i), alpha=0.6, marker='^', facecolors='none' , s=80)
        i += 1
    ax.grid()
    ax.legend(prop={'size':18})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel("$k$", fontsize=20)
    ax.set_title("K dimension", fontsize=20)
    
    plt.savefig(save_path, dpi=300)


def construct_km_profiler(raw_path, save_path, k_list, m_list, system="MP-SPDZ", analyze_path=None, repeat=50, degree=2):
    """Construct the km_profiler for different MPC systems.

    Args:
        raw_path (str): The file path for the raw time record.
        save_path (str): The file path for km_profiler.pkl stored.
        system (str, optional): Target MPC systems, setting for different log structures. 
            Defaults to "MP-SPDZ".
    """
    y = []
    x = []
    
    if system == "MP-SPDZ":
        for k in k_list:
            for m in m_list:
                x.append([m, k])

        with open(raw_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[:7] == "Stopped":
                    segs = line.split(" ")
                    y.append(float(segs[-1]))
    
    x = np.array(x)
    y = np.array(y) / repeat
    
    km_profiler = SubPoly(degree=2, fit_intercept=True)
    km_profiler.fit(x, y)
    km_profiler.model_analysis(x, y)
    
    p = open(save_path, 'wb')
    pickle.dump(km_profiler, p)
    print("----> Save model in ", save_path)
    
    if analyze_path is not None:
        profiler_analysis(k_list, m_list, x, y, km_profiler.predict(x), analyze_path)
        
    
    
    
    

    
