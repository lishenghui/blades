from typing import Optional, Generator, Dict

import torch
import torch.nn as nn
import numpy as np
from .client import BladesClient

j = 1


class RSAClient(BladesClient):
    def __init__(
        self,
        per_model: nn.Module,
        per_opt: torch.optim.Optimizer,
        lambda_: float,
        id: Optional[str] = None,
        momentum: Optional[float] = 0.0,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        super(RSAClient, self).__init__(id=id, momentum=momentum, device=device)
        self._per_model = per_model
        self._per_opt = per_opt
        self.lambda_ = lambda_

    def train_global_model(
        self, train_set: Generator, num_batches: int, opt: torch.optim.Optimizer
    ) -> None:
        pass

    def get_personal_model(self) -> nn.Module:
        return self._per_model

    def train_personal_model(
        self, train_set: Generator, num_batches: int, global_state: Dict
    ) -> None:
        r"""Local optimizaiton of the ``client``. Byzantine input can override
        this method to perform adversarial attack.

        Args:
            data_batches: A list of training batches for local training.
            opt: Optimizer.
        """
        self._per_model.train()
        global j
        for i in range(num_batches):
            data, target = next(train_set)
            data, target = data.to(self.device), target.to(self.device)
            self._per_opt.zero_grad()

            output = self._per_model(data)
            # Clamp loss value to avoid possible 'Nan' gradient with some
            # attack types.
            loss = torch.clamp(self.loss_func(output, target), 0, 1e6)
            loss.backward()

            for name, p in self._per_model.named_parameters():
                sign = torch.sign(global_state[name] - p)

                p.grad.data += torch.Tensor(self.lambda_ * sign / np.sqrt(j + 1))
            j += 1
            self._per_opt.step()
