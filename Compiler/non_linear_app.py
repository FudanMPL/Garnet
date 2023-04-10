import json
from Compiler import floatingpoint
from Compiler.types import *
from Compiler.oram import *

def getFile(path):
    with open(path, "r") as f:
        js = json.load(f)
        return js

@vectorize
def nlf_app(func, x):
    # 读取多项式
    content = getFile("./Compiler/NFA/%s.json"%(func))
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


