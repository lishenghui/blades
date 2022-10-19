import ray
from blades.core.actor import Actor
from ray.util import ActorPool
import torch
from torch.multiprocessing.reductions import reduce_tensor
from torch.nn import Module
from blades.datasets.fldataset import FLDataset
from blades.utils.torch_utils import get_num_params
from blades.utils.collective import setup_dist
from blades.servers import BladesServer
from blades.clients import BladesClient
import torch.distributed as dist
from typing import Dict, List, TypeVar
from blades.utils.torch_utils import parameters_to_vector
from torch.optim import Optimizer
import logging
import numpy as np

# import os
import copy

T = TypeVar("T", bound="Optimizer")
T_SER = TypeVar("T_SER", bound="BladesServer")

logger = logging.getLogger(__name__)


@ray.remote
class ActorManager:
    def __init__(
        self,
        dataset: FLDataset,
        global_model: Module,
        opt_cls: T,
        opt_kws: Dict,
        num_actors: int,
        num_buffers: 0,
        gpu_per_actor: float,
        world_size: int = 0,
        rank: int = None,
        server_cls: T_SER = None,
        server_kws: Dict = None,
        num_selected_clients: int = None,
        device: str = "cpu",
        visible_gpu: str = "0",
    ):
        """_summary_

        Args:
            dataset (FLDataset): _description_
            global_model (Module): _description_
            opt_cls (T): _description_
            opt_kws (Dict): _description_
            num_actors (int): _description_
            gpu_per_actor (float): _description_
            world_size (int, optional): _description_. Defaults to 0.
            rank (int, optional): _description_. Defaults to 0.
            server (BladesServer, optional): _description_. Defaults to None.
        """

        self.rank = rank
        self.device = device
        self.world_size = world_size
        self.num_selected_clients = num_selected_clients
        self.num_params = get_num_params(global_model)
        block_groups = np.array_split(range(num_buffers), num_actors)
        self.shared_memory = torch.zeros((num_buffers, self.num_params)).to(self.device)
        self.shared_memory[
            0,
        ] = parameters_to_vector(global_model.parameters()).detach()
        self.mem_meta_info = reduce_tensor(self.shared_memory)

        if server_cls:
            server_kws |= {
                "shared_memory": self.shared_memory,
                # "mem_meta_info": self.mem_meta_info,
                "device": self.device,
            }
            self.server = server_cls(**server_kws)

        self.ray_actors = [
            Actor.options(num_gpus=gpu_per_actor).remote(
                dataset,
                global_model,
                opt_cls,
                opt_kws,
                self.mem_meta_info,
                list(block_groups[i]),
            )
            for i in range(num_actors)
        ]
        ray.get([actor.init.remote() for actor in self.ray_actors])
        self.actor_pool = ActorPool(self.ray_actors)

    def init(self):
        return True

    def init_dist(self):
        world_size = self.world_size
        rank = self.rank
        if world_size > 0:
            self.group = setup_dist(world_size, rank)
            if self.rank == 0:
                self.gather_list = [self.shared_memory] * world_size

    def get_mem_meta_info(self):
        return self.mem_meta_info

    def train(
        self,
        clients: List[BladesClient],
        # global_model: Module,
        local_round: int = 1,
        lr: float = 1.0,
    ):
        """_summary_

        Args:
            clients (List[BladesClient]): _description_
            global_model (Module): _description_
            local_round (int): _description_
            lr (float): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        self.broadcast()

        client_groups = np.array_split(clients, len(self.ray_actors))
        result_ids = []
        for clients, actor in zip(client_groups, self.ray_actors):
            ref_clients = actor.local_train.remote(
                clients=clients,
                num_rounds=local_round,
                # lr=lr,
            )
            result_ids.append(ref_clients)
        while len(result_ids):
            _, result_ids = ray.wait(result_ids)

        # breakpoint()
        if self.world_size <= 1:
            self.server.global_update()

        else:
            # pass
            self.gather()
        return True

    def evaluate(
        self,
        clients: List[BladesClient],
        # global_model: Module,
        round_number: int = 1,
        metrics: Dict = None,
    ):
        """_summary_

        Args:
            clients (List[BladesClient]): _description_
            global_model (Module): _description_
            round_number (int): _description_
            lr (float): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        # dst = 0
        # if self.world_size > 0:
        #     if self.rank == 0:
        #         self.broadcast_buffer = parameters_to_vector(
        #             self.server.model.parameters()
        #         )
        #         dist.broadcast(tensor=self.broadcast_buffer, src=self.rank)
        #     else:
        #         dist.broadcast(tensor=self.shared_memory[0], src=dst)

        client_groups = np.array_split(clients, len(self.ray_actors))
        results = []
        for clients, actor in zip(client_groups, self.ray_actors):
            ret_id = actor.evaluate.remote(
                clients=clients, round_number=round_number, metrics=metrics
            )
            ret = ray.get(ret_id)
            results.extend(ret)
        return results

    def broadcast(self):
        dst = 0
        if self.world_size > 0:
            if self.rank == 0:
                self.shared_memory[0] = parameters_to_vector(
                    self.server.model.parameters()
                )
                dist.broadcast(tensor=self.shared_memory[0], src=self.rank)
            else:
                dist.broadcast(tensor=self.shared_memory[0], src=dst)

    def gather(self):
        dst = 0
        if self.rank == 0:
            # gather_memory = copy.deepcopy(self.shared_memory.detach())
            # breakpoint()
            dist.gather(tensor=self.shared_memory, gather_list=self.gather_list)
            # dist.gather(tensor=self.shared_memory, gather_list=self.gather_list)
            # breakpoint()
            updates = torch.cat(self.gather_list)
            # breakpoint()
            self.server.global_update(updates)
            # self.broadcast_buffer = parameters_to_vector(self.server.model.parameters())
        else:
            # pass
            dist.gather(tensor=self.shared_memory, dst=dst)
