import logging
from typing import Dict, List, TypeVar

import numpy as np
import ray
import torch
import torch.distributed as dist
from ray.util import ActorPool
from torch.multiprocessing.reductions import reduce_tensor

from blades.clients import BladesClient
from blades.core.actor import Actor
from blades.datasets.fldataset import FLDataset
from blades.models import get_model
from blades.utils.collective import setup_dist
from blades.utils.torch_utils import get_num_params, parameters_to_vector
from blades.utils.utils import set_random_seed

# import torch

T = TypeVar("T", bound="Optimizer")
T_SER = TypeVar("T_SER", bound="BladesServer")

logger = logging.getLogger(__name__)


@ray.remote
class ActorManager:
    def __init__(
        self,
        dataset: FLDataset,
        model: str,
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

        set_random_seed()
        model_tmp = get_model(model).to("cuda")
        self.num_params = get_num_params(model_tmp)
        block_groups = np.array_split(range(num_buffers), num_actors)
        self.shared_memory = torch.zeros((num_buffers, self.num_params)).to(self.device)
        self.shared_memory[
            0,
        ] = parameters_to_vector(model_tmp.parameters()).detach()
        self.mem_meta_info = reduce_tensor(self.shared_memory)
        # print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))

        if server_cls:
            server_kws |= {
                "model": model_tmp,
                "shared_memory": self.shared_memory,
                # "mem_meta_info": self.mem_meta_info,
                "device": self.device,
            }
            self.server = server_cls(**server_kws)

        self.ray_actors = [
            Actor.options(num_gpus=gpu_per_actor).remote(
                dataset,
                model,
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
                self.gather_list = [
                    torch.zeros_like(self.shared_memory) for _ in range(world_size)
                ]

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

        if self.world_size <= 1:
            updates = self.shared_memory

        else:
            updates = self.gather()

        if hasattr(self, "server"):
            for idx, client in enumerate(self.server.get_clients()):
                client.save_update(updates[idx])
            for idx, client in enumerate(self.server.get_clients()):
                if client.is_byzantine():
                    client.omniscient_callback(self.server)

            self.server.global_update()

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
            dist.gather(tensor=self.shared_memory, gather_list=self.gather_list)
            updates = torch.cat(self.gather_list)
            return updates
        else:
            dist.gather(tensor=self.shared_memory, dst=dst)
