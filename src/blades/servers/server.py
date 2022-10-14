from typing import Callable, List

from blades.clients import BladesClient
import torch
import torch.distributed as dist
from blades.utils.torch_utils import get_num_params
from blades.utils.torch_utils import parameters_to_vector
from blades.utils.collective import setup_dist
import ray


@ray.remote
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
        world_size: int = 0,
    ):
        self.optimizer = optimizer
        self.model = model
        self.aggregator = aggregator

        # Enable `torch.distributed` if GPU is available
        if self.__ray_metadata__.num_gpus > 0:
            num_params = get_num_params(model)
            self.group = setup_dist(world_size, world_size)
            self.gather_list = [torch.zeros(num_params)] * world_size
            self.broad_cast_buffer = torch.zeros(num_params)

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
        server_rank = 0
        if hasattr(self, "group"):
            h = dist.gather(tensor=self.broad_cast_buffer, gather_list=self.gather_list)
            h.wait()

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

        if hasattr(self, "group"):
            self.broad_cast_buffer = parameters_to_vector(self.model.parameters())
            h = dist.broadcast(tensor=self.broad_cast_buffer, src=server_rank)
            h.wait()
