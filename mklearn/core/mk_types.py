import typing

import numpy as np
import torch
from .learn_module import LearnModule

NDArray = typing.TypeVar('NDArray', bound=np.ndarray)
NDTensor = typing.TypeVar("NDTensor", bound=torch.Tensor)
SwitchDeviceAble = typing.Union[torch.device, torch.Tensor, torch.nn.Module, LearnModule, str, None]
CalculationScalar = typing.Union[
    int, float,
    np.int, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float32, np.float, np.float64, np.double,
    torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    torch.float, torch.float32, torch.float64, torch.double,
    torch.complex, torch.complex32, torch.complex64, torch.complex128,
    torch.cfloat, torch.cdouble
]
__all__ = [
    "NDArray",
    "NDTensor",
    "SwitchDeviceAble"
]
