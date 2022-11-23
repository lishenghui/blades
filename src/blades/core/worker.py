import copy
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import ray
import torch
import torch.nn as nn

from blades.datasets.fldataset import FLDataset
from blades.models import get_model
from blades.utils.torch_utils import get_num_params
from blades.utils.torch_utils import vector_to_parameters
from blades.utils.utils import set_random_seed
from .communicator import Communicator


# from blades.utils.torch_utils import parameters_to_vector, vector_to_parameters

# from .dist_actor import BaseActor
# T = TypeVar("T", bound="Optimizer")


@ray.remote
class Worker(Communicator):
    """Ray Worker."""

    def __init__(
        self,
        dataset: FLDataset,
        # model: nn.Module,
        model_name: str,
        opt_cls,
        opt_kws: Dict,
        *,
        clients: List = None,
        # mem_meta_info: tuple = None,
        # buffer_blocks: List[int] = None,
        seed: Optional[int] = 0,
    ):
        """_summary_

        Args:
            dataset (FLDataset): _description_
            id (int): _description_
            mem_meta_info (tuple): _description_
            model (nn.Module): _description_
            opt (torch.optim.Optimizer): _description_
            lr (float): _description_
        #
        """
        # super(Worker, self).__init__()
        super().__init__()
        # self.device = "cpu" if ray.get_gpu_ids() == [] else "cuda"
        set_random_seed(seed)
        self.dataset = dataset
        self.model = get_model(model_name).to(self._device)
        self.optimizer = opt_cls(self.model.parameters(), **opt_kws)
        if clients is not None:
            self.clients = clients

        client_rank = 0
        for client in self.clients:
            client.set_local_rank(client_rank)
            client_rank += 1

        self.random_states = {}
        self._num_model_params = get_num_params(self.model)

    def get_num_clients(self):
        return len(self.clients)

    def get_num_model_params(self):
        return self._num_model_params

    def cache_random_state(self) -> None:
        # This function should be used for reproducibility
        if ray.get_gpu_ids() != []:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()
        self.random_states["python"] = random.getstate()

    def restore_random_state(self) -> None:
        # This function should be used for reproducibility
        if ray.get_gpu_ids() != []:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])
        random.setstate(self.random_states["python"])

    def init(self):
        return True

    # def create_shared_memory(self, length):
    #     if self.local_rank() == 0:
    #         self.shared_memory = torch.zeros((length, self.num_params)).to(
    #         self.device)
    #         self.shared_memory[
    #             0,
    #         ] = parameters_to_vector(self.model.parameters()).detach()
    #         self.mem_meta_info = reduce_tensor(self.shared_memory)
    #         return self.get_gpu_id(), self.mem_meta_info
    #     else:
    #         return None, None
    #
    # def init_dist(self, mem_dic, world_size, dst_rank):
    #     if self.get_gpu_id() != -1:
    #         mem_meta_info = mem_dic[self.get_gpu_id()]
    #         self.world_size = world_size
    #         self.dst_rank = dst_rank
    #         if self.dst_rank == self.get_gpu_id():
    #             self.base_mem = (self.local_rank() - 1) * len(self.clients)
    #         else:
    #             self.base_mem = self.local_rank() * len(self.clients)
    #         if self.local_rank() == 0:
    #             self.group = setup_dist(world_size, self.get_gpu_id())
    #         else:
    #             self.shared_memory = mem_meta_info[0](*mem_meta_info[1])
    #     else:
    #         pass
    #         self.group = setup_dist(
    #             world_size,
    #         )

    def set_lr(self, lr: float) -> None:
        r"""change the learning rate of the client optimizer.

        Args:
            lr (float): target learning rate.
        """
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def local_train(
        self,
        # clients: List[BladesClient],
        *,
        num_rounds: int = 1,
        global_model: nn.Module = None,
    ) -> List:
        """A proxy method that provides local training for a set of clients.

        Args:
            clients (List): a list of clients.
            num_rounds (int, optional): number of local steps. Defaults to 1.
            global_model (nn.Module, optional): the global global_model from server. \
                                                Defaults to None.

        Returns:
            List: a list of the given clients.
        """
        clients = self.clients
        # self.cache_random_state()
        if not global_model:
            model_vec = copy.deepcopy(self.shared_memory[0])
        for client in clients:
            if global_model:
                self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            else:
                vector_to_parameters(copy.deepcopy(model_vec), self.model.parameters())

            client.set_global_model_ref(self.model)
            local_dataset = self.dataset.get_train_loader(client.id())
            client.train_global_model(
                train_set=local_dataset, num_batches=num_rounds, opt=self.optimizer
            )
            client.train_personal_model(
                train_set=local_dataset,
                num_batches=num_rounds,
                global_state=self.model.state_dict(),
            )
            self.shared_memory[
                self.get_local_rank() + client.get_local_rank()
            ] = client.get_update()

        # if self.get_local_rank() == 0:
        #     breakpoint()
        # update = torch.stack(list(map(lambda w: w.get_update(), clients)))

        # breakpoint()
        # self.shared_memory[
        #     base_mem : base_mem + len(self.clients),
        # ] = update
        # self.restore_random_state()

        return clients

    def evaluate(
        self,
        round_number: int = None,
        batch_size: int = 128,
        metrics=None,
    ):
        update = []
        vector_to_parameters(self.shared_memory[0], self.model.parameters())
        # breakpoint()
        self.model.eval()
        for client in self.clients:
            client.set_global_model_ref(self.model)
            data = self.dataset.get_all_test_data(client.id())
            result = client.evaluate(
                round_number=round_number,
                test_set=data,
                batch_size=batch_size,
                metrics=metrics,
            )
            update.append(result)
        return update

    # def gather(self):
    #     # dst = 0
    #     # if self.global_rank() == 0:
    #     #     dist.gather(tensor=self.shared_memory, gather_list=self.gather_list)
    #     #     updates = torch.cat(self.gather_list)
    #     #     return updates
    #     if self.local_rank() == 0 and self.get_gpu_id() != -1:
    #         dist.gather(tensor=self.shared_memory, dst=self.dst_rank)
    #     else:
    #         return
    #
    # def broadcast(self):
    #     if self.local_rank() == 0 and self.world_size > 1:
    #         dist.broadcast(tensor=self.shared_memory[0], src=self.dst_rank)


def _assign_global_ranks(server, actors: List[Worker]):
    ser_submitted = server.set_global_rank.remote(0)

    actors_submitted = [
        actor.set_global_rank.remote(i + 1) for i, actor in enumerate(actors)
    ]
    ray.get([ser_submitted] + actors_submitted)


def assign_rank(server, actors: List[Worker]):
    _assign_global_ranks(server, actors)

    ser_rank = ray.get(server.get_global_rank.remote())
    worker_ranks = ray.get([actor.get_global_rank.remote() for actor in actors])
    num_clients = ray.get([actor.get_num_clients.remote() for actor in actors])
    num_clients_mapping = defaultdict(lambda: 0)
    for idx, num in zip(worker_ranks, num_clients):
        num_clients_mapping[idx] += num

    gpu_mapping = defaultdict(lambda: 0)
    local_ranks = []
    rets = []

    for g_id, actor in zip(worker_ranks, actors):
        current_rank = gpu_mapping[g_id]
        gpu_mapping[g_id] += 1
        ret = actor.set_local_rank.remote(current_rank)
        local_ranks.append(current_rank)
        rets.append(ret)

    ray.get(rets)

    shared_mem_len = max(num_clients_mapping.values())
    num_model_param = ray.get(actors[0].get_num_model_params.remote())
    # actors = [
    #     x
    #     for x, _, _ in sorted(
    #         zip(actors + [server], worker_ranks + [ser_rank], local_ranks + [0]),
    #         key=lambda x: (x[1], x[2]),
    #     )
    # ]
    actors = actors + [server]
    mem_meta_info = dict(
        (x, y)
        for x, y in ray.get(
            [
                actor.create_shared_memory.remote((shared_mem_len, num_model_param))
                for actor in actors
            ]
        )
        if x is not None
    )
    print(mem_meta_info.keys())
    world_size = max(worker_ranks) + 1
    submitted = [
        actor.setup_dist.remote(mem_meta_info, world_size, ser_rank) for actor in actors
    ]
    ret = ray.get(submitted)
