from typing import Callable, List

import torch
import numpy as np
from blades.clients import BladesClient
from .server import BladesServer

i = 1


class RSAServer(BladesServer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        aggregator: Callable[[list], torch.Tensor],
        l1_lambda: float = 0.07,
        weight_lambda: float = 0.01,
        batches_per_round: int = 1,
    ):

        super(RSAServer, self).__init__(optimizer, model, aggregator=aggregator)
        self.l1_lambda = l1_lambda
        self.weight_lambda = weight_lambda
        self.batches_per_round = batches_per_round

    def global_update(self, clients: List[BladesClient]) -> None:
        r"""Apply a step of global optimization.

            .. note::
                The input should be a ``Tensor``, which will be converted to
                ``pseudo-gradient`` layer by layer.

        Args:
            update: The aggregated update.
        """
        global i
        self.zero_grad()
        for name, p in self.model.named_parameters():
            tmp = torch.mean(
                torch.stack(
                    list(
                        map(
                            lambda w: torch.sign(
                                p - w.get_personal_model().state_dict()[name]
                            ),
                            clients,
                        )
                    )
                ),
                dim=0,
            )
            p.grad = (self.l1_lambda * tmp + self.weight_lambda * p) / np.sqrt(i + 1)
        i += 1
        self.optimizer.step()
        return
