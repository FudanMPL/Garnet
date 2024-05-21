import Compiler.nn as nn

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
    nn.Conv2d,
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    nn.Linear,
)
