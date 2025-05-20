import sys
import os
sys.path.append(os.environ.get('GARNET_HOME', ''))
from Compiler.compilerLib import Compiler
from Compiler.types import *
from Compiler.library import for_range
from Compiler.Convert.model import ConvertModel
from Compiler.Convert import onnxSpilter
from Compiler.tensor import Tensor, reset_gloabal_store, reset_op_id
import onnx
import subprocess
import torch
import contextlib
import json

db = {}
log_file = open('ts.log', 'w')

def runMPL(onnx_model):
    compiler = Compiler()
    @compiler.register_function('test_onnx')
    def test_onnx():
        model = ConvertModel(onnx_model)

        x = MultiArray(in_size, sfix)
        @for_range(x.total_size())
        def _(i):
            x.assign_vector(0.01, i)

        input = Tensor(x, req_grad = True)
        y = model(input)
        
        reset_op_id()
        reset_gloabal_store()
        
    with contextlib.redirect_stdout(log_file):
        compiler.compile_func()
    
    result = str(compiler.prog.req_num)
    try:
        round = int(re.search(r"(\d+) online round", result).group(1))
        comm = int(re.search(r"(\d+) online communication bits", result).group(1))
        offround = int(re.search(r"(\d+) offline round", result).group(1))
        offcomm = int(re.search(r"(\d+) offline communication bits", result).group(1))
        return round, comm, offround, offcomm
    except Exception as e:
        return 0, 0, 0, 0

def profiling(model_path, input_size):
    round_list = []
    comm_list = []
    offround_list = []
    offcomm_list = []
    
    models, info_list = onnxSpilter.split_model_by_node(model_path)
    for m, info in zip(models, info_list):
        key = (re.sub(r'\d+$', '', info[0]),) + info[1:]
        key_str = json.dumps(key)
        input_size = tuple(key[1][0])
        print(key_str, input_size)
        if key_str in db:
            round, comm, offround, offcomm = db[key_str]
        else: 
            global in_size
            in_size = input_size
            round, comm, offround, offcomm = runMPL(m)
            db[key_str] = (round, comm, offround, offcomm)
        
        print((round, comm, offround, offcomm))
        round_list.append(round)
        comm_list.append(comm)
        offround_list.append(offround)
        offcomm_list.append(offcomm)
    
    return sum(round_list), sum(comm_list), sum(offround_list), sum(offcomm_list)

if __name__=='__main__':
    # profiling("example.onnx", input_size=(1,32,10,10))
    profiling("resnet50.onnx", input_size=(1,64,56,56))
    
    