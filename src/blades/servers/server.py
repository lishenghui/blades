from typing import Callable, Dict

import ray
import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor

from blades.clients import BladesClient
from blades.models import get_model
from blades.utils.collective import setup_dist
from blades.utils.torch_utils import get_num_params
from blades.utils.torch_utils import parameters_to_vector
from blades.utils.utils import reset_model_weights, set_random_seed

# T = TypeVar("T", bound="Optimizer")


@ray.remote
class BladesServer(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        model: torch.nn.Module,
        clients: BladesClient,
        opt_cls=torch.optim.SGD,
        opt_kws: Dict = None,
        aggregator: Callable[[list], torch.Tensor] = None,
        world_size: int = 0,
        device: str = "cuda",
        random_seed=0,
        # mem_meta_info: torch.Tensor = None,
        # shared_memory: torch.Tensor = None,
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
        # if mem_meta_info:
        #     self.shared_memory = mem_meta_info[0](*mem_meta_info[1])
        # else:
        #     self.shared_memory = shared_memory
        self.clients = clients
        self.device = device
        set_random_seed(random_seed)
        self.model = get_model(model).to(self.device)

        reset_model_weights(self.model)
        self.num_params = get_num_params(self.model)
        # self.model = model().to("cuda")
        self.optimizer = opt_cls(self.model.parameters(), **opt_kws)
        self.aggregator = aggregator

        # Enable `torch.distributed` if GPU is available
        # if world_size > 0:
        #     num_params = get_num_params(model)
        #     self.group = setup_dist(world_size, 0)
        #     self.gather_list = [torch.zeros(num_params).to(self.device)] * world_size
        #     self.broad_cast_buffer = torch.zeros(num_params).to(self.device)

    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]

    def local_rank(self):
        return self._lcoal_rank

    def global_rank(self):
        return self._global_rank

    def set_ranks(self, global_rank, local_rank):
        self._global_rank = global_rank
        self._lcoal_rank = local_rank

    def create_shared_memory(self, length):
        self.shared_memory = torch.zeros((length, self.num_params)).to(self.device)
        self.shared_memory[
            0,
        ] = parameters_to_vector(self.model.parameters()).detach()
        self.mem_meta_info = reduce_tensor(self.shared_memory)
        return self.get_gpu_id(), self.mem_meta_info

    def gather(self):
        # dst = 0
        # if self.global_rank() == 0:
        dist.gather(
            tensor=self.shared_memory,
            gather_list=self.gather_list,
            dst=self.get_gpu_id(),
        )
        # self.updates = torch.cat(self.gather_list)
        # print(updates)
        # breakpoint()
        # return updates
        # elif self.local_rank() == 0:
        #     dist.gather(tensor=self.shared_memory, dst=dst)
        # else:
        #     return

    def broadcast(self):
        self.shared_memory[0] = parameters_to_vector(self.model.parameters())
        if self.world_size > 1:
            dist.broadcast(tensor=self.shared_memory[0], src=self.get_gpu_id())

    def init_dist(self, mem_dic, world_size, ser_gpu_id):
        mem_meta_info = mem_dic[self.get_gpu_id()]
        self.world_size = world_size
        # rank = self._global_rank
        if self.local_rank() == 0:
            self.group = setup_dist(world_size, ser_gpu_id)
        else:
            self.shared_memory = mem_meta_info[0](*mem_meta_info[1])

        if world_size > 0:
            self.gather_list = [
                torch.zeros_like(self.shared_memory) for _ in range(world_size)
            ]

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

    def global_update(self, update_list=None) -> None:
        r"""Apply a step of global optimization.

            .. note::
                The input should be a ``Tensor``, which will be converted to
                ``pseudo-gradient`` layer by layer.

        Args:
            update: The aggregated update.
        #
        """
        # if update_list is not None:
        #     self.gather_list = update_list
        # else:
        #     self.gather_list = self.clients
        # self.gather_list = self.shared_memory
        updates = torch.cat(self.gather_list)
        updates = updates[: len(self.clients), :]
        # print("updates from clients", updates)
        # updates = torch.cat(self.gather_list)[:len[self.clients],]
        grad = self.aggregator(updates)
        # breakpoint()
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
        model_vec = parameters_to_vector(self.model.parameters())
        self.shared_memory[
            0,
        ] = model_vec
        # print("new model", model_vec)
        return True
