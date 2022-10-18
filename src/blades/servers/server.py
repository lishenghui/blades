from typing import Callable, List, Dict, TypeVar

from blades.clients import BladesClient
import torch
import torch.distributed as dist
from blades.utils.torch_utils import get_num_params
from blades.utils.torch_utils import parameters_to_vector
from blades.utils.collective import setup_dist
from torch.optim import Optimizer
import ray


T = TypeVar("T", bound="Optimizer")


# @ray.remote
class BladesServer(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        model: torch.nn.Module,
        opt_cls: T = torch.optim.SGD,
        opt_kws: Dict = None,
        aggregator: Callable[[list], torch.Tensor] = None,
        world_size: int = 0,
        device: str = "cpu",
        mem_meta_info: torch.Tensor = None,
        shared_memory: torch.Tensor = None,
    ):
        """_summary_

        Args:
            model (torch.nn.Module): _description_
            opt_cls (T): _description_
            opt_kws (Dict): _description_
            aggregator (Callable[[list], torch.Tensor]): _description_
            world_size (int, optional): _description_. Defaults to 0.
            device (str, optional): _description_. Defaults to
             "cpu".
            mem_meta_info (torch.Tensor, optional): _description_. Defaults to None.
        """
        if mem_meta_info:
            self.shared_memory = mem_meta_info[0](*mem_meta_info[1])
        else:
            self.shared_memory = shared_memory
        self.model = model
        self.optimizer = opt_cls(self.model.parameters(), **opt_kws)
        self.aggregator = aggregator
        self.device = device
        # Enable `torch.distributed` if GPU is available
        if world_size > 0:
            num_params = get_num_params(model)
            self.group = setup_dist(world_size, 0)
            self.gather_list = [torch.zeros(num_params).to(self.device)] * world_size
            self.broad_cast_buffer = torch.zeros(num_params).to(self.device)

    def get_opt(self) -> torch.optim.Optimizer:
        """Returns the global optimizer.

        Returns:
            torch.optim.Optimizer: _description_
        """
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

    def global_update(
        self, clients: List[BladesClient] = None, update_list=None
    ) -> None:
        r"""Apply a step of global optimization.

            .. note::
                The input should be a ``Tensor``, which will be converted to
                ``pseudo-gradient`` layer by layer.

        Args:
            update: The aggregated update.
        """
        if update_list:
            self.gather_list = update_list
        else:
            self.gather_list = self.shared_memory

        update = self.aggregator(self.gather_list)
        # print(update)
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
        model_vec = parameters_to_vector(self.model.parameters())
        self.shared_memory[
            0,
        ] = model_vec
        return True
