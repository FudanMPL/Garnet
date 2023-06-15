from Compiler import floatingpoint
from Compiler.types import *
from Compiler.oram import *

import json
import sys
import os

sys.path.append("Compiler/GFA/")
from NFGen.main import generate_nonlinear_config
import NFGen.PerformanceModel.time_ops as to

platform = "Rep3" # using MP-SPDZ Rep3 protocol as an example.
profiler_file = 'Compiler/GFA/NFGen/PerformanceModel/' + platform + "_kmProfiler.pkl"

def getFile(path):
    with open(path, "r") as f:
        js = json.load(f)
        return js

def getFuncName(func):
    return str(func).split(" ")[1]

def f2p(func, kmax, f, n, range):

    config = {
        "function": func, # function config.
        "range": range,
        "k_max": kmax, # set the maximum order.
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
    kmconfig = generate_nonlinear_config(config)

    polysFile = "./Compiler/GFA/%s.json"%(getFuncName(func))
    with open(polysFile, "w") as f:
        json.dump(kmconfig, f, indent=4)

@vectorize
def At(x, breaks, coeffA, scaler):
    # m 段多项式 每段 k 次 最高次为degree
    m = len(coeffA)
    k = len(coeffA[0])
    degree = k-1
            
    # 预计算 x 的 1 到 degree 次方
    pre_muls = floatingpoint.PreOpL(lambda a,b,_: a * b, [x] * degree)
            
    # 计算出每段上 x 对应的函数值
    poss_res = sfix.Array(m)
    for i in range(m):
        poss_res[i] = coeffA[i][0] * scaler[i][0]
        for j in range(degree):
            poss_res[i] += coeffA[i][j+1] * pre_muls[j] * scaler[i][j+1]
            
    # 将 x 与断点依次比较
    comp = sfix.Array(m)
    for i in range(m):
        comp[i] = (x >= breaks[i])

    # 保留最后一个比较成功的位置为 1 ，其它置回 0
    cipher_index = sfix.Array(m)
    @for_range_opt(m-1)
    def f(i):
        cipher_index[i] = comp[i+regint(1)]
    @for_range_opt(m)
    def f(i):
        cipher_index[i] = comp[i]*(comp[i] - cipher_index[i])

    # 计算每段函数值向量与位置向量的点积
    return sfix.dot_product(cipher_index, poss_res)

def GFA(kmax=10, f=31, n=63, range=(-10,10)):
    def x(func):
        def y(x):
            fname = getFuncName(func)
            fpath = "./Compiler/GFA/%s.json"%(fname)
            # translating
            if not os.path.exists(fpath):
                f2p(func, kmax, f, n, range)
            # loading
            polys = getFile(fpath)
            breaks = polys['breaks']
            coeffA = polys['coeffA']
            scaler = polys['scaler']
            # caculating
            return At(x, breaks, coeffA, scaler)

        return y

    return x


# (2.0 version deleted)
'''
class GFA:
    def __init__(self, func):
        # 读取多项式
        polys = getFile("./Compiler/GFA/%s.json"%(func))
        breaks = polys['breaks']
        coeffA = polys['coeffA']
        scaler = polys['scaler']
    
    @vectorize
    def At(self, x):
        # m 段多项式 每段 k 次 最高次为degree
        m = len(coeffA)
        k = len(coeffA[0])
        degree = k-1
        
        # 预计算 x 的 1 到 degree 次方
        pre_muls = floatingpoint.PreOpL(lambda a,b,_: a * b, [x] * degree)
        
        # 计算出每段上 x 对应的函数值
        poss_res = sfix.Array(m)
        for i in range(m):
            poss_res[i] = coeffA[i][0] * scaler[i][0]
            for j in range(degree):
                poss_res[i] += coeffA[i][j+1] * pre_muls[j] * scaler[i][j+1]
        
        # 将 x 与断点依次比较
        comp = sfix.Array(m)
        for i in range(m):
            comp[i] = (x >= breaks[i])

        # 保留最后一个比较成功的位置为 1 ，其它置回 0
        cipher_index = sfix.Array(m)
        @for_range_opt(m-1)
        def f(i):
            cipher_index[i] = comp[i+regint(1)]
        @for_range_opt(m)
        def f(i):
            cipher_index[i] = comp[i]*(comp[i] - cipher_index[i])

        # 计算每段函数值向量与位置向量的点积
        return sfix.dot_product(cipher_index, poss_res)
'''

# (1.0 version deleted)
'''
@vectorize
def nlf_app(func, x):
    # 读取多项式
    content = getFile("./Compiler/GFA/%s.json"%(func))
    breaks = content['breaks']
    coeffA = content['coeffA']
    scaler = content['scaler']

    # m 段多项式 每段 k 次 最高次为degree
    m = len(coeffA)
    k = len(coeffA[0])
    degree = k-1
    
    # 预计算 x 的 1 到 degree 次方
    pre_muls = floatingpoint.PreOpL(lambda a,b,_: a * b, [x] * degree)
    
    # 计算出每段上 x 对应的函数值
    poss_res = sfix.Array(m)
    for i in range(m):
        poss_res[i] = coeffA[i][0] * scaler[i][0]
        for j in range(degree):
            poss_res[i] += coeffA[i][j+1] * pre_muls[j] * scaler[i][j+1]
    
    # 将 x 与断点依次比较
    comp = sfix.Array(m)
    for i in range(m):
        comp[i] = (x >= breaks[i])

    # 保留最后一个比较成功的位置为 1 ，其它置回 0
    cipher_index = sfix.Array(m)
    @for_range_opt(m-1)
    def f(i):
        cipher_index[i] = comp[i+regint(1)]
    @for_range_opt(m)
    def f(i):
        cipher_index[i] = comp[i]*(comp[i] - cipher_index[i])

    # 计算每段函数值向量与位置向量的点积
    return sfix.dot_product(cipher_index, poss_res)
'''