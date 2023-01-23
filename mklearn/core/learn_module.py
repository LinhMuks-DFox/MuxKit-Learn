import abc
from typing import *

import torch.nn


class LearnModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.device_ = "cpu"

    @abc.abstractmethod
    def to_device(self, device: Union[str, torch.device]):
        pass

    @abc.abstractmethod
    def summary(self) -> str:
        pass

    @abc.abstractmethod
    def properties_dict(self, **kwargs) -> Dict[str, Any]:
        pass

    def __str__(self):
        return self.summary()

    __repr__ = __str__

    def predict(self, x_sample):
        return self.forward(x_sample)

    def to(self,
           device: Optional[Union[int, torch.device]] = None,
           dtype: Optional[Union[torch.dtype, str]] = None,
           non_blocking: bool = False) -> 'torch.Module':
        self.device_ = device if device is not None else self.device_
        return super().to(device, dtype, non_blocking)
