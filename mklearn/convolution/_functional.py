from typing import *

import torch.nn as nn

from ._ConvNDimNTimes import ConvNDimNTimes
from ._ConvTransposeNDimNTimes import ConvTransposeNDimNTimes


def conv_to_transpose(cov: ConvNDimNTimes, inverse_conv_property: bool = True) -> ConvTransposeNDimNTimes:
    if not isinstance(cov, ConvNDimNTimes):
        raise TypeError("Can not build ConvTransposeNDimNTimes instance from a non-ConvNDimNTimes instance.")
    raise NotImplementedError("Not implemented yet.")


def transpose_to_conv(transpose: ConvTransposeNDimNTimes) -> ConvNDimNTimes:
    raise NotImplementedError("Not implemented yet.")


def new_conv_conv_transpose_pair() -> Tuple[ConvNDimNTimes, ConvTransposeNDimNTimes]:
    raise NotImplementedError("Not implemented yet.")


def new_conv_linear_adapter(conv: ConvNDimNTimes, *args, **kwargs) -> nn.Module:
    raise NotImplementedError("Not implemented yet.")
