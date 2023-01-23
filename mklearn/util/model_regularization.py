from typing import Dict, Any, Union

import torch

from mklearn.core.learn_module import LearnModule


class LNRegularization(LearnModule):

    def __init__(self, n_regular: int = 2, n_lambda: float = 0.001, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.n_regular_ = n_regular
        self.n_lambda_ = n_lambda  # weight
        self.to_device(device)

    def forward(self, loss_fn_res, model: Union[torch.Module, LearnModule]) -> torch.Tensor:
        ln_parameters = torch.sum(torch.tensor(
            [torch.pow(param, self.n_regular_)
             for param in model.parameters()]
        ))
        return loss_fn_res + self.n_lambda_ * ln_parameters

    def to_device(self, device: Union[str, torch.device]):
        self.device_ = torch.device(device) if isinstance(device, str) else self.device
        self.to(self.device_)

    def summary(self) -> str:
        return str(self.properties_dict())

    def properties_dict(self, **kwargs) -> Dict[str, Any]:
        return {
            "n_regular_": self.n_regular_,
            "n_lambda_": self.n_lambda_,
        }


class L1Regularization(LNRegularization):

    def __init__(self, n_lambda: float = 0.001, device: Union[str, torch.device] = "cpu"):
        super().__init__(n_regular=1, n_lambda=n_lambda, device=device)

    def forward(self, loss_fn_res, model: Union[torch.Module, LearnModule]) -> torch.Tensor:
        l1_parameters = torch.sum(torch.tensor(
            [torch.abs(param)
             for param in model.parameters()]
        ))
        return loss_fn_res + self.n_lambda_ * l1_parameters
