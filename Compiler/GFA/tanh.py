import sympy as sp
from NFGen.main import generate_nonlinear_config
# import NFGen.CodeTemplet.templet as temp
import NFGen.PerformanceModel.time_ops as to

platform = "Rep3" # using MP-SPDZ Rep3 protocol as an example.
f = 31
n = 63
profiler_file = './NFGen/PerformanceModel/' + platform + "_kmProfiler.pkl"

# fundenmental functions, indicating they are cipher-text non-linear operations.
def func_reciprocal(x):
        return 1 / x

def func_exp(x, lib=sp):
    return lib.exp(x)


# target function.
def tanh(x):
    ep = func_exp(x)
    en = func_exp(-x)
    return (ep - en) * func_reciprocal(ep + en)

# define NFD
sigmoid_config = {
    "function": tanh, # function config.
    "nickname": "tanh",
    "range": (-50, 50),
    "k_max": 10, # set the maximum order.
    "tol": 1e-3, # percision config.
    "ms": 1000, # maximum samples.
    "zero_mask": 1e-6,
    "n": n, # <n, f> fixed-point config.
    "f": f,
    "profiler": profiler_file, # profiler model source file.
    # "code_templet": temp.templet_spdz, # spdz templet.
    # "code_language": "python", # indicating the templet language.
    # "config_file": "./sigmoig_spdz.py", # generated code file.
    "time_dict": to.basic_time[platform], # basic operation time cost.
    # "test_graph": "./graph/" # (optional, need mkdir for this folder first), whether generate the graph showing the approximation and the real function.
}

# using NFGen to generate the target function code.
generate_nonlinear_config(sigmoid_config)