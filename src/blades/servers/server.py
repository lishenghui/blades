from typing import Callable, List

from blades.clients import BladesClient
import torch


class BladesServer(object):
    r"""Simulating the server of the federated learning system.

    :ivar aggregator: a callable which takes a list of tensors and returns
            an aggregated tensor.
    :vartype aggregator: callable

    :param  optimizer: The global optimizer, which can be any optimizer
    from Pytorch.
    :type optimizer: torch.optim.Optimizer
    :param model: The global global_model
    :type model: torch.nn.Module
    :param aggregator: a callable which takes a list of tensors and returns
            an aggregated tensor.
    :type aggregator: callable
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        aggregator: Callable[[list], torch.Tensor],
    ):
        self.optimizer = optimizer
        self.model = model
        self.aggregator = aggregator

    def get_opt(self) -> torch.optim.Optimizer:
        r"""Returns the global optimizer."""
        return self.optimizer

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.
        It should be called before assigning pseudo-gradient.

        Args:
            set_to_none: See `Pytorch documentation <https://pytorch.org/docs/s
            table/generated/torch.optim.Optimizer.zero_grad.html>`_.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_model(self) -> torch.nn.Module:
        r"""Returns the current global global_model."""
        return self.model

    def global_update(self, clients: List[BladesClient]) -> None:
        r"""Apply a step of global optimization.

            .. note::
                The input should be a ``Tensor``, which will be converted to
                ``pseudo-gradient`` layer by layer.

        Args:
            update: The aggregated update.
        """
        update = self.aggregator(clients)
        self.zero_grad()
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                end = beg + len(p.data.view(-1))
                x = update[beg:end].reshape_as(p.data)
                p.grad = -x.clone().detach().to(p.device)
                beg = end
        self.optimizer.step()
