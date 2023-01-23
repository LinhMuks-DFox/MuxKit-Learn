import torch
from mklearn.core.mk_types import *
from mklearn.core.learn_module import LearnModule
from mklearn.core.mklearn_errors import InvalidComparison


def is_same_device(left: SwitchDeviceAble,
                   right: SwitchDeviceAble, ) -> bool:
    print(f"left: {left}, right: {right}")

    if left is None or right is None:
        return False
    if isinstance(left, torch.Tensor):
        left = left.device
    elif isinstance(left, (torch.nn.Module, LearnModule)):
        left = next(left.parameters()).device
    elif isinstance(left, str):
        left = torch.device(left)
    elif isinstance(left, torch.device):
        pass
    else:
        raise InvalidComparison(f"Cannot compare left with type: {type(left)}")

    if isinstance(right, torch.Tensor):
        right = right.device
    elif isinstance(right, (torch.nn.Module, LearnModule)):
        right = next(right.parameters()).device
    elif isinstance(right, str):
        right = torch.device(right)
    elif isinstance(right, torch.device):
        pass
    else:
        raise InvalidComparison(f"Cannot compare right with type: {type(right)}")
    # print(f"left: {left}, right: {right}")
    return left == right