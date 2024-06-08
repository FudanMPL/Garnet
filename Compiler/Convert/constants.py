import Compiler.nn as nn
# from Compiler.nn.modules.conv import _ConvNd
from onnx2pytorch.operations import (
    BatchNormWrapper,
    InstanceNormWrapper,
    Loop,
    LSTMWrapper,
    Split,
    TopK,
)

from Compiler.Convert.Ops import split

COMPOSITE_LAYERS = (nn.Sequential,)
# MULTIOUTPUT_LAYERS = (nn.MaxPool2d, Loop, LSTMWrapper, split.Split, TopK)
MULTIOUTPUT_LAYERS = (Loop, LSTMWrapper, split.Split, TopK)
STANDARD_LAYERS = (
    # _ConvNd,
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    nn.Linear,
)
