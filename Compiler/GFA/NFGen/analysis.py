import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from NFGen.funcs import calculate_decimal, calculate_zero, piece_prediction, sampled_error, analytic_error, convert_function


def analytic_error_analysis(folder, func_name, k, x, breaks, coeffA, scalerA, target_func, f, n, zero_mask=None, method="absolute_error"):
    """Generate the analytic metric analysis figures.
    """
    m = len(breaks)

    # save the function graph
    exec_func = convert_function(target_func)
    y_pred = piece_prediction(x[:, np.newaxis], coeffA, breaks, scalerA, f, n)
    y_true = exec_func(x)
    breaks_label = piece_prediction(np.array(breaks)[:, np.newaxis], coeffA, breaks, scalerA, f, n)

    p1 = plt.plot(x, y_true, 'r--', label='Rel')
    p2 = plt.plot(x, y_pred, 'b--', label='Pred')
    plt.plot(breaks, breaks_label, 'o', label='Break points')
    plt.legend()
    plt.title("%s-k=%d, m=%d"%(func_name, k, m))
    plt.savefig(folder+func_name+"-(k,m=%d %d).png"%(k, m))
    plt.clf()

    # calculate the analytic error for each bin.
    error_list = []
    for i in range(m-1):
        coeff = coeffA[i]
        scaler = scalerA[i]
        interval_coeff = coeff * scaler
        a, b = breaks[i], breaks[i+1]

        max_error = analytic_error(target_func, interval_coeff, a, b, f, zero_mask=zero_mask, method=method)
        error_list.append(max_error)

    x_pos = np.arange(m-1)
    fig, ax = plt.subplots()
    ax.bar(x_pos, error_list, align='center', capsize=3, alpha=0.7)
    plt.ylabel("Analytic Max Error")
    ax.set_xticks(x_pos)
    ax.set_title("Max error of %s"%(func_name))
    plt.tight_layout()
    ax.yaxis.grid(True)

    for a, b in zip(x_pos, error_list):
        ax.text(a, b, '%.2g'%b, ha="center", va="bottom",fontsize=11)

    plt.savefig(folder+func_name+"-Max-Error-(k,m=%d %d).png"%(k, m))
    plt.clf()
    return 


def sampled_error_analysis(folder, func_name, k, x, y_true, y_pred, breaks, breaks_val, f, zero_mask=None, method="relative_error"):
    """Generate the sampled metric analysis figures.
    """
    m = len(breaks)

    # save the function graph
    p1 = plt.plot(x, y_true, 'r--', label='Rel')
    p2 = plt.plot(x, y_pred, 'b--', label='Pred')
    plt.plot(breaks, breaks_val, 'o', label='Break points')
    plt.legend()
    plt.title("%s-k=%d, m=%d"%(func_name, k, m))
    plt.savefig(folder+func_name+"-(k,m=%d %d).png"%(k, m))
    # plt.savefig('./graph/%s-(k=%d, m=%d).png'%(func, k, m))
    plt.clf()

    # save the error graph
    x_group = [[] for _ in range(m)]
    each_error = sampled_error(y_true, y_pred, f, zero_mask=zero_mask, method=method)
    error_group = [[] for _ in range(m)]
    for i in range(len(x)):
        index = np.sum(x[i] >= breaks) - 1
        x_group[index].append(x[i])
        error = each_error[index]
        error_group[index].append(error)

    mean_list = []
    std_list = []

    for i in range(m):
        mean_list.append(np.mean(error_group[i]))
        std_list.append(np.std(error_group[i]))

    x_pos = np.arange(m)

    fig, ax = plt.subplots()
    ax.bar(x_pos, mean_list, yerr=std_list, align='center', capsize=3, alpha=0.7)
    ax.set_ylabel("Predict Error")
    ax.set_xticks(x_pos)
    ax.set_title("Group error of %s"%(func_name))
    plt.tight_layout()
    ax.yaxis.grid(True)

    plt.savefig(folder+func_name+"Error-bar-(k,m=%d %d).png"%(k, m))
    plt.clf()

    return


def generate_config(file_name, func_name, coeffA, breaks, scaler):
    """Using to generate the config file for cipher-text computation.
    """
    with open(file_name, 'a') as f:
        f.write(func_name+"_config = {\n")
    
        # write breaks
        string_breaks = "    \'breaks\': " + "["
        for i in range(len(breaks)-1):
            string_breaks += str(breaks[i]) + ", "
        string_breaks += str(breaks[-1]) + '], \n'
        f.write(string_breaks)
        
        # write scaler_list
        string_scaler = "    \'scaler\': " + "[\n"
        f.write(string_scaler)
        for i in range(len(breaks)-1):
            each_line = "    ["
            for j in range(len(scaler[0])):
                each_line += str(scaler[i][j]) + ", "
            each_line += "], \n"
            f.write(each_line)
        f.write("    ], \n")
        
        # write coeffA
        string_config = "    \'coeffA\': " + "[\n"
        f.write(string_config)
        for i in range(len(breaks)-1):
            each_line = "    ["
            for j in range(len(coeffA[0])):
                each_line += str(coeffA[i][j]) + ", "
            each_line += "], \n"
            f.write(each_line)
        f.write("    ], \n")
        f.write("}, \n\n")
        
    print("Write config for %s function SUCCESS!!"%(func_name))
    

def macro_generation(config, func_name, file_name="./cpp_config.txt"):
    """Generate the macro config for cpp applications.
    """
    breaks = config['breaks']
    scaler = config['scaler']
    coeffA = config['coeffA']
    
    shape = (len(scaler), len(scaler[0]))
    
    with open(file_name, 'a') as f:
        f.write("#define "+ func_name +"_K "+str(shape[1])+'\n')
        f.write("#define "+ func_name +"_M "+str(shape[0])+'\n')

        f.write("#define "+ func_name +"_BREAKS {")
        for i in range(len(breaks)):
            if(i < len(breaks)-1):
                f.write(str(breaks[i])+", ")
            else:
                f.write(str(breaks[i]))
        f.write("}\n")

        # write scaler_list
        string_scaler = "#define "+ func_name +"_SCALER {"
        f.write(string_scaler)
        for i in range(len(breaks)-1):
            each_line = ""
            for j in range(len(scaler[0])):
                if(j<len(scaler[0])-1):
                    each_line += str(scaler[i][j]) + ", "
                else:
                    each_line += str(scaler[i][j])
            each_line += ", "
            f.write(each_line)
        f.write("} \n")


        # write config
        string_coeff = "#define "+ func_name +"_COEFF {"
        f.write(string_coeff)
        for i in range(len(breaks)-1):
            each_line = ""
            for j in range(len(coeffA[0])):
                if(j<len(coeffA[0])-1):
                    each_line += str(coeffA[i][j]) + ", "
                else:
                    each_line += str(coeffA[i][j])
            each_line += ", "
            f.write(each_line)
        f.write("}\n\n")
        
    print("Write config for %s function SUCCESS!!"%(func_name))


def save_fig(folder, func_name, k, x, y_true, y_pred, breaks, breaks_val, f, zero_mask=None):
    """Save the fitting fig.
    """
    m = len(breaks)

    # save the function graph
    p1 = plt.plot(x, y_true, 'r--', label='Rel')
    p2 = plt.plot(x, y_pred, 'b--', label='Pred')
    plt.plot(breaks, breaks_val, 'o', label='Break points')
    plt.legend()
    plt.title("%s-k=%d, m=%d"%(func_name, k, m))
    plt.savefig(folder+func_name+"-(k,m=%d %d).png"%(k, m))
    # plt.savefig('./graph/%s-(k=%d, m=%d).png'%(func, k, m))
    plt.clf()

    # save the test error bar
    x_group = [[] for _ in range(m)]
    print(">>>> IN TEST <<<<< ")
    each_error = relative_error(y_true, y_pred, f, zero_mask=zero_mask)
    error_group = [[] for _ in range(m)]
    for i in range(len(x)):
        index = np.sum(x[i] >= breaks) - 1
        x_group[index].append(x[i])
        # error = relative_error(y_true[index], y_pred[index], f)
        error = each_error[index]
        error_group[index].append(error)

    mean_list = []
    std_list = []

    for i in range(m):
        mean_list.append(np.mean(error_group[i]))
        std_list.append(np.std(error_group[i]))

    x_pos = np.arange(m)

    fig, ax = plt.subplots()
    ax.bar(x_pos, mean_list, yerr=std_list, align='center', capsize=3, alpha=0.7)
    ax.set_ylabel("Predict Error")
    ax.set_xticks(x_pos)
    ax.set_title("Group error of %s"%(func_name))
    plt.tight_layout()
    ax.yaxis.grid(True)

    plt.savefig(folder+func_name+"Error-bar-(k,m=%d %d).png"%(k, m))
    plt.clf()

    return


def eff_cal(b_item, item):
    """Calculate the efficiency changing.
    gain = 1 - (item / b_item)
    """
    return 1 - (item / b_item)


def analysis_efficiency(func_list, benchmark, test, func, km_list):
    """Generate pandas analyzing the efficiency.
    """
    dict = {func_list[i]: [km_list[i], benchmark[i], test[i], func(benchmark[i], test[i])] for i in range(len(func_list))}
    df = pd.DataFrame(dict)
    df.index = ['KM', 'Benchmark', 'Our Method', 'Gain']
    
    return df