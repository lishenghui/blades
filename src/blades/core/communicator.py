import copy
import warnings
from collections import defaultdict
from typing import List
from typing import Tuple

import ray
import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor

from blades.utils.collective import setup_dist
from blades.utils.torch_utils import vector_to_parameters


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
        self._model_vec = None

    def init(self):
        return True

    def set_local_rank(self, local_rank):
        self._local_rank = local_rank

    def get_local_rank(self):
        return self._local_rank

    def set_global_rank(self, rank: int):
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids != []:
            warnings.warn(
                "`global_rank` is meant to `gpu_id` if cuda is enable. "
                f"Setting it to `{gpu_ids[0]}`."
            )
            rank = gpu_ids[0]
        self._global_rank = rank
        return self._global_rank

    def set_memo_idx(self, base, length):
        self._memo_base = base
        self._memo_length = length

    def save_update(self, rank, update_vec):
        self.shared_memory[self._memo_base + rank] = update_vec

    def get_global_rank(self):
        if self._global_rank is None:
            warnings.warn("`global_rank` is `None` so far, have you explicitly set?")
        return self._global_rank

    def create_shared_memory(self, shape: Tuple):
        if self._is_communicator():
            self.shared_memory = torch.zeros(shape).to(self._device)
            self.mem_meta_info = reduce_tensor(self.shared_memory)

            self.model_memo = torch.zeros(shape[1]).to(self._device)
            self.model_mem_meta = reduce_tensor(self.model_memo)
            return self.get_global_rank(), (self.mem_meta_info, self.model_mem_meta)
        else:
            return None, None

    def setup_dist(self, mem_dic, world_size, dst_rank):
        self.world_size = world_size
        self._dis_rank = dst_rank

        if self._is_communicator():
            self.group = setup_dist(
                world_size, self.get_global_rank(), backend=self._backend
            )
        else:
            mem_meta_info = mem_dic[self.get_global_rank()][0]
            self.shared_memory = mem_meta_info[0](*mem_meta_info[1])

            model_mem_meta_info = mem_dic[self.get_global_rank()][1]
            self.model_memo = model_mem_meta_info[0](*model_mem_meta_info[1])

        if self.get_global_rank() == self._dis_rank:
            self.gather_list = [
                torch.zeros_like(self.shared_memory) for _ in range(world_size)
            ]

    def _is_communicator(self):
        return self.get_local_rank() == 0

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

    def get_valid_updates(self):
        assert self.get_global_rank() == self._dis_rank
        if self._memo_length == len(self.gather_list[0]):
            updates = torch.cat(self.gather_list[1:])
        else:
            updates = torch.cat(self.gather_list)[self._memo_length :, :]
        return updates

    def broadcast(self):
        if self._is_communicator():
            dist.broadcast(tensor=self.model_memo, src=self._dis_rank)

    def load_model_from_memory(self, model):
        vector_to_parameters(copy.deepcopy(self.model_memo), model.parameters())


def _assign_global_ranks(server, actors: List[Communicator]):
    ser_submitted = server.set_global_rank.remote(0)

    actors_submitted = [
        actor.set_global_rank.remote(i + 1) for i, actor in enumerate(actors)
    ]
    return ray.get([ser_submitted] + actors_submitted)


def assign_rank(server, actors: List[Communicator]):
    global_ranks = _assign_global_ranks(server, actors)
    world_size = len(set(global_ranks))

    ser_rank = ray.get(server.get_global_rank.remote())
    all_actors = [server] + actors
    num_clients = [0] + ray.get([actor.get_num_clients.remote() for actor in actors])

    num_clients_mapping = defaultdict(lambda: 0)
    for idx, num in zip(global_ranks, num_clients):
        num_clients_mapping[idx] += num

    gpu_mapping = defaultdict(lambda: 0)
    memo_base_mapping = defaultdict(lambda: 0)
    local_ranks = []
    rets = []

    shared_mem_len = max(num_clients_mapping.values())
    num_model_param = ray.get(actors[0].get_num_model_params.remote())

    for memo_length, g_id, actor in zip(num_clients, global_ranks, all_actors):
        current_rank = gpu_mapping[g_id]
        gpu_mapping[g_id] += 1
        ret1 = actor.set_local_rank.remote(current_rank)
        if g_id == ser_rank and global_ranks.count(ser_rank) < 2:
            memo_length = shared_mem_len
        memo_base = memo_base_mapping[g_id]
        print(memo_base)
        ret2 = actor.set_memo_idx.remote(memo_base, memo_length)
        memo_base_mapping[g_id] += memo_length
        local_ranks.append(current_rank)
        rets.extend([ret1, ret2])

    ray.get(rets)

    mem_meta_info = dict(
        (x, y)
        for x, y in ray.get(
            [
                actor.create_shared_memory.remote((shared_mem_len, num_model_param))
                for actor in all_actors
            ]
        )
        if x is not None
    )
    submitted = [
        actor.setup_dist.remote(mem_meta_info, world_size, ser_rank)
        for actor in all_actors
    ]
    ray.get(submitted)
