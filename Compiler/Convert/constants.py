import nn as nn

from onnx2pytorch.operations import (
    BatchNormWrapper,
    InstanceNormWrapper,
    Loop,
    LSTMWrapper,
    Split,
    TopK,
)


COMPOSITE_LAYERS = (nn.Sequential,)
MULTIOUTPUT_LAYERS = (nn.MaxPool2d, Loop, LSTMWrapper, Split, TopK)
STANDARD_LAYERS = (
    nn.Conv2d,
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    nn.Linear,
)
