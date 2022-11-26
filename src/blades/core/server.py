from typing import Callable, Dict, List

import ray
import torch
import torch.distributed as dist

from blades.clients import BladesClient
from blades.models import get_model
from blades.utils.torch_utils import get_num_params
from blades.utils.torch_utils import parameters_to_vector
from blades.utils.utils import reset_model_weights, set_random_seed
from .communicator import Communicator


# T = TypeVar("T", bound="Optimizer")


@ray.remote
class BladesServer(Communicator):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        model: torch.nn.Module,
        clients: List[BladesClient],
        opt_cls=torch.optim.SGD,
        opt_kws: Dict = None,
        aggregator: Callable[[list], torch.Tensor] = None,
        random_seed=0,
    ):
        """_summary_

        Args:
            model (torch.nn.Module): _description_
            opt_cls (T): _description_
            opt_kws (Dict): _description_
            aggregator (Callable[[list], torch.Tensor]): _description_
            world_size (int, optional): _description_. Defaults to 0.
            _device (str, optional): _description_. Defaults to
             "cpu".
            mem_meta_info (torch.Tensor, optional): _description_. Defaults to None.
        """
        # if mem_meta_info:
        #     self.shared_memory = mem_meta_info[0](*mem_meta_info[1])
        # else:
        #     self.shared_memory = shared_memory
        super().__init__()
        self.clients = clients
        self.device = "cpu" if ray.get_gpu_ids() == [] else "cuda"
        set_random_seed(random_seed)
        self.model = get_model(model).to(self.device)

        reset_model_weights(self.model)
        self.num_params = get_num_params(self.model)
        # self.model = model().to("cuda")
        self.optimizer = opt_cls(self.model.parameters(), **opt_kws)
        self.aggregator = aggregator
        # self.set_local_rank()

    def get_clients(self):
        return self.clients

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

    def broadcast(self):
        model_vec = parameters_to_vector(self.model.parameters())
        dist.broadcast(tensor=model_vec, src=self._dis_rank)

    def global_update(self, update_list=None) -> None:
        r"""Apply a step of global optimization.

            .. note::
                The input should be a ``Tensor``, which will be converted to
                ``pseudo-gradient`` layer by layer.

        Args:
            update: The aggregated update.
        #
        """
        updates = self.get_valid_updates()
        grad = self.aggregator(updates)
        self.zero_grad()
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                end = beg + len(p.data.view(-1))
                x = grad[beg:end].reshape_as(p.data)
                p.grad = -x.clone().detach().to(p.device)
                beg = end
        self.optimizer.step()
