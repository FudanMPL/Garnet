import Compiler.nn as nn
import Compiler.functional as F
from Compiler.tensor import Tensor

from onnx2pytorch.operations.base import Operator

from typing import Union, Tuple, Any, Callable, Iterable,Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

class Conv2d(Operator):
    def __init__(self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        

    def forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, groups = self.groups)

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
