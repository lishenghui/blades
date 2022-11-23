import warnings
from typing import Tuple
from typing import List
import ray
import torch
import torch.distributed as dist
from collections import defaultdict
from torch.multiprocessing.reductions import reduce_tensor

from blades.utils.collective import setup_dist


class Communicator(object):
    """Base Actor."""

    def __init__(
        self,
    ):
        self._device = "cpu" if ray.get_gpu_ids() == [] else "cuda"
        self._backend = "gloo" if self._device == "cpu" else "nccl"
        self._local_rank = -1
        self._global_rank = None
        self._mem_base = 0
        self._dis_rank = 0

    def set_local_rank(self, local_rank):
        self._local_rank = local_rank

    def get_local_rank(self):
        return self._local_rank

    def set_global_rank(self, rank: int):
        # breakpoint()
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids != []:
            warnings.warn(
                "`global_rank` is meant to `gpu_id` if cuda is enable."
                f"Setting it to `{gpu_ids[0]}`."
            )
            rank = gpu_ids[0]
        self._global_rank = rank

    def save_update(self, rank, update_vec):
        self.shared_memory[self.get_local_rank() + rank] = update_vec

    def get_global_rank(self):
        if self._global_rank is None:
            warnings.warn("`global_rank` is `None` so far, have you explicitly set?")
        return self._global_rank

    def create_shared_memory(self, shape: Tuple):
        if self._is_communicator():
            self.shared_memory = torch.zeros(shape).to(self._device)
            self.mem_meta_info = reduce_tensor(self.shared_memory)
            return self.get_global_rank(), self.mem_meta_info
        else:
            return None, None

    def _is_communicator(self):
        if self.get_global_rank() == self._dis_rank and self.get_local_rank() < 0:
            return True
        elif self.get_global_rank() != self._dis_rank and self.get_local_rank() == 0:
            return True
        else:
            return False

    def setup_dist(self, mem_dic, world_size, dst_rank):
        self.world_size = world_size
        self._dis_rank = dst_rank

        mem_meta_info = mem_dic[self.get_global_rank()]
        if self._is_communicator():
            self.group = setup_dist(
                world_size, self.get_global_rank(), backend=self._backend
            )

        self.shared_memory = mem_meta_info[0](*mem_meta_info[1])
        if self.get_global_rank() == self._dis_rank:
            self.gather_list = [
                torch.zeros_like(self.shared_memory) for _ in range(world_size)
            ]

    def gather(self):
        if self._is_communicator():
            if self.get_global_rank() == self._dis_rank:
                dist.gather(
                    tensor=self.shared_memory,
                    gather_list=self.gather_list,
                    dst=self._dis_rank,
                )
            else:
                dist.gather(tensor=self.shared_memory, dst=self._dis_rank)
        else:
            return

    def broadcast(self):
        if self._is_communicator():
            dist.broadcast(tensor=self.shared_memory[0], src=self._dis_rank)


def _assign_global_ranks(server, actors: List[Communicator]):
    ser_submitted = server.set_global_rank.remote(0)

    actors_submitted = [
        actor.set_global_rank.remote(i + 1) for i, actor in enumerate(actors)
    ]
    ray.get([ser_submitted] + actors_submitted)


def assign_rank(server, actors: List[Communicator]):
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
    actors = [
        x
        for x, _, _ in sorted(
            zip(actors + [server], worker_ranks + [ser_rank], local_ranks + [0]),
            key=lambda x: (x[1], x[2]),
        )
    ]
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
    world_size = max(worker_ranks) + 1
    submitted = [
        actor.setup_dist.remote(mem_meta_info, world_size, ser_rank) for actor in actors
    ]
    ray.get(submitted)
