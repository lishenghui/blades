import ray
from blades.core.actor import _RayActor
from ray.util import ActorPool
import torch
from typing import List
from torch.multiprocessing.reductions import reduce_tensor
from torch.nn import Module
from blades.datasets.fldataset import FLDataset
from blades.utils.torch_utils import get_num_params
from blades.utils.collective import setup_dist
from blades.servers import BladesServer
from blades.clients import BladesClient
import torch.distributed as dist


@ray.remote
class ActorManager:
    def __init__(
        self,
        global_model: Module,
        num_actors: int,
        gpu_per_actor: float,
        dataset: FLDataset,
        world_size: int = 0,
        rank: int = 0,
        server: BladesServer = None,
    ):
        if server:
            self.server = server
        num_params = get_num_params(global_model)
        self.shared_memory = torch.randn((num_actors, num_params))
        mem_meta_info = reduce_tensor(self.shared_memory)
        self.ray_actors = [
            _RayActor.options(num_gpus=gpu_per_actor).remote(dataset, i, mem_meta_info)
            for i in range(num_actors)
        ]
        self.actor_pool = ActorPool(self.ray_actors)

        # Enable `torch.distributed` if GPU is available
        if self.__ray_metadata__.num_gpus > 0:
            self.group = setup_dist(world_size, rank)

    def train(
        self,
        clients: List[BladesClient],
        global_model: Module,
        local_round: int,
        lr: float,
    ):
        server_rank = 0
        result_ids = []
        for clients, actor in zip(clients, self.ray_actors):
            ref_clients = actor.local_training.remote(
                clients=clients,
                global_model=global_model,
                local_round=local_round,
                lr=lr,
            )
            result_ids.append(ref_clients)
        while len(result_ids):
            _, result_ids = ray.wait(result_ids)

        if hasattr(self, "server"):
            raise NotImplementedError

        elif hasattr(self, "group"):
            h = dist.gather(tensor=self.shared_memory, dst=server_rank)
            h.wait()
            h = dist.broadcast(tensor=self.shared_memory, src=server_rank)
            h.wait()
            return True
