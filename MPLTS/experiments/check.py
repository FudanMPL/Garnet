import sys
import os
sys.path.append(os.environ.get('GARNET_HOME', ''))
from Compiler.compilerLib import Compiler
from Compiler.types import *
from Compiler.library import for_range
from Compiler.Convert.model import ConvertModel
from Compiler.tensor import Tensor, reset_gloabal_store, reset_op_id
import onnx
import subprocess
import torch


def runPlain(model_path):
    from onnx2pytorch import ConvertModel
    onnx_model = onnx.load(model_path)
    pytorch_model = ConvertModel(onnx_model)
    
    x = torch.full((1, 32, 10, 10), 0.01)
    y = pytorch_model(x)
    print(y)

in_size = (1,32,10,10)

def runMPL(model_path):
    compiler = Compiler()
    @compiler.register_function('test_onnx')
    def test_onnx():
        onnx_model = onnx.load(model_path)
        model = ConvertModel(onnx_model)

        x = MultiArray(in_size, sfix)
        @for_range(x.total_size())
        def _(i):
            x.assign_vector(0.01, i)

        input = Tensor(x, req_grad = True)
        y = model(input)
        
        y.value.print_reveal_nested()
        
        reset_op_id()
        reset_gloabal_store()
    
    compiler.compile_func()
    
    print("compile %s finished, press enter to continue..."%(model_path))
    input()
    
    response = subprocess.run(['Scripts/semi2k.sh', 'test_onnx'], check=True)
    
    print("compile %s finished, press enter to continue..."%(model_path))
    input()
    # response = subprocess.run(['Scripts/semi2k.sh', '-F', 'test_onnx'], check=True)

if __name__=='__main__':
    runPlain("example.onnx")
    runMPL("example.onnx")
    runMPL("example_opt.onnx")