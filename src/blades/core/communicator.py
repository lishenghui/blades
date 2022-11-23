import warnings
from typing import Tuple

import ray
import torch
import torch.distributed as dist
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

    def set_local_rank(self, local_rank):
        self._local_rank = local_rank
        print(f"`local rank`: {self._local_rank}")

    def get_local_rank(self):
        return self._local_rank

    # def set_mem_base(self, base):
    #     self._mem_base = base
    #
    # def get_mem_base(self):
    #     return self._mem_base

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

        print(f"`global rank`: {self._global_rank}")

    def get_global_rank(self):
        if self._global_rank is None:
            warnings.warn("`global_rank` is `None` so far, have you explicitly set?")
        return self._global_rank

    def create_shared_memory(self, shape: Tuple):
        print(f"{self.get_global_rank()} creating memory")
        if self.get_local_rank() <= 0:
            self.shared_memory = torch.zeros(shape).to(self._device)
            self.mem_meta_info = reduce_tensor(self.shared_memory)
            return self.get_global_rank(), self.mem_meta_info
        else:
            print("Creating nothing..............")
            return None, None

    def setup_dist(self, mem_dic, world_size, dst_rank):
        self.world_size = world_size
        self.dst_rank = dst_rank

        mem_meta_info = mem_dic[self.get_global_rank()]
        # breakpoint()
        if self.get_local_rank() <= 0:
            self.group = setup_dist(
                world_size, self.get_global_rank(), backend=self._backend
            )

        self.shared_memory = mem_meta_info[0](*mem_meta_info[1])
        # self.base_mem = 0
        if self.get_global_rank() == self.dst_rank:
            self.gather_list = [
                torch.zeros_like(self.shared_memory) for _ in range(world_size)
            ]

    def gather(self):
        if self.get_local_rank() <= 0:
            if self.get_global_rank() == self.dst_rank:
                dist.gather(
                    tensor=self.shared_memory,
                    gather_list=self.gather_list,
                    dst=self.dst_rank,
                )
            else:
                dist.gather(tensor=self.shared_memory, dst=self.dst_rank)
        else:
            return

    def broadcast(self):
        if self.get_local_rank() <= 0 and self.world_size > 1:
            dist.broadcast(tensor=self.shared_memory[0], src=self.dst_rank)
