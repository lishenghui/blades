from blades.core import ActorManager
import torch
import ray
from blades.servers import BladesServer
from blades.clients import BladesClient
import logging
import sys
from blades.datasets.fldataset import FLDataset
from typing import Dict, List, TypeVar
from torch.optim import Optimizer

# from blades.utils.torch_utils import parameters_to_vector
from tqdm import trange
import numpy as np
from blades.utils.utils import (
    initialize_logger,
    top1_accuracy,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

T = TypeVar("T", bound="Optimizer")
T_SER = TypeVar("T_SER", bound="BladesServer")


class Simulator(object):
    def __init__(
        self,
        dataset: FLDataset,
        clients: List[BladesClient],
        num_gpus: int = 0,
        num_gpus_mgr: float = 0.2,
        num_actors_mgr: int = 5,
        num_gpus_actor: float = 0.15,
        local_opt_cls: T = None,
        local_opt_kws: Dict = None,
        global_model: torch.nn.Module = None,
        server_cls: T_SER = None,
        server_kws: Dict = None,
        device: str = "cuda",
        log_path: str = "./outputs",
    ) -> None:

        self.act_mgrs = []
        self.num_gpus = num_gpus
        self.clients = clients
        num_clients = len(clients)
        client_groups = np.array_split(range(num_clients), num_gpus)

        for i in range(num_gpus):
            if i == 0:
                server_cls = server_cls
                server_kws = server_kws
                server_kws["clients"] = clients
                num_gpus_mgr = 0.2
            else:
                server_cls = None
                num_gpus_mgr = 0.2

            actor_mgr = ActorManager.options(num_gpus=num_gpus_mgr).remote(
                dataset,
                global_model,
                local_opt_cls,
                local_opt_kws,
                rank=i,
                num_actors=num_actors_mgr,
                num_buffers=len(client_groups[i]),
                num_selected_clients=num_clients,
                gpu_per_actor=num_gpus_actor,
                world_size=num_gpus,
                server_cls=server_cls,
                server_kws=server_kws,
                device=device,
            )
            ray.get(actor_mgr.init.remote())
            self.act_mgrs.append(actor_mgr)

        initialize_logger(log_path)
        self.json_logger = logging.getLogger("stats")
        ray.get([mgr.init_dist.remote() for mgr in self.act_mgrs])

    def run(self, validate_interval: int = 100, global_rounds: int = 4000):
        with trange(0, global_rounds + 1) as t:
            for global_rounds in t:
                client_groups = np.array_split(self.clients, self.num_gpus)
                ret_ids = [
                    actor_mgr.train.remote(clients=client_group)
                    for (actor_mgr, client_group) in zip(self.act_mgrs, client_groups)
                ]
                ray.get(ret_ids)
                if global_rounds % validate_interval == 0 and global_rounds > 0:
                    ray.get([mgr.broadcast.remote() for mgr in self.act_mgrs])
                    ret_ids = [
                        actor_mgr.evaluate.remote(
                            clients=client_group,
                            round_number=global_rounds,
                            metrics={"top1": top1_accuracy},
                        )
                        for (actor_mgr, client_group) in zip(
                            self.act_mgrs, client_groups
                        )
                    ]

                    results = ray.get(ret_ids)
                    results = [item for sublist in results for item in sublist]
                    test_results = self.log_validate(results)
                    t.set_postfix(loss=test_results[0], top1=test_results[1])

    def log_validate(self, metrics):
        top1 = np.average(
            [metric["top1"] for metric in metrics],
            weights=[metric["Length"] for metric in metrics],
        )
        loss = np.average(
            [metric["Loss"] for metric in metrics],
            weights=[metric["Length"] for metric in metrics],
        )
        r = {
            "_meta": {"type": "test"},
            "Round": metrics[0]["E"],
            "top1": top1,
            "Length": np.sum([metric["Length"] for metric in metrics]),
            "Loss": loss,
        }
        self.json_logger.info(r)
        return loss, top1
