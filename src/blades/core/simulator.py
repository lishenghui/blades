import logging
import sys
from typing import Dict, List, Optional

import numpy as np
import ray
import torch
import wandb
from tqdm import trange

from blades.clients import BladesClient
from blades.core.worker import Worker
from blades.core.communicator import assign_rank
from blades.datasets.fldataset import FLDataset
from blades.utils.utils import top1_accuracy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# T = TypeVar("T", bound="Optimizer")
# T_SER = TypeVar("T_SER", bound="BladesServer")


class Simulator(object):
    def __init__(
        self,
        dataset: FLDataset,
        clients: List[BladesClient],
        num_gpus: int = 0,
        num_actors: int = 5,
        num_gpus_actor: float = 0.15,
        local_opt_cls=None,
        local_opt_kws: Dict = None,
        global_model: torch.nn.Module = None,
        server_cls=None,
        server_kws: Dict = None,
        num_gpus_server: Optional[float] = 0.1,
        device: str = "cuda",
        log_path: str = "./outputs",
    ) -> None:

        self.act_mgrs = []
        self.num_gpus = num_gpus
        self.clients = clients
        client_groups = np.array_split(self.clients, num_actors)

        server_cls = server_cls
        server_kws = server_kws
        server_kws["clients"] = clients

        self.ray_actors = [
            Worker.options(num_gpus=num_gpus_actor).remote(
                dataset,
                global_model,
                local_opt_cls,
                local_opt_kws,
                clients=client_groups[i],
            )
            for i in range(num_actors)
        ]
        ray.get([actor.init.remote() for actor in self.ray_actors])

        server_kws |= {
            "model": global_model,
            "clients": self.clients,
        }
        self.server = server_cls.options(num_gpus=num_gpus_server).remote(**server_kws)

        assign_rank(self.server, self.ray_actors)

        # initialize_logger(log_path)
        # self.json_logger = logging.getLogger("stats")

    def run(self, validate_interval: int = 100, global_rounds: int = 4000):
        with trange(0, global_rounds + 1) as t:
            for global_rounds in t:
                ray.get(
                    [
                        actor.broadcast.remote()
                        for actor in self.ray_actors + [self.server]
                    ]
                )
                ray.get([actor.local_train.remote() for actor in self.ray_actors])

                all_ret = [
                    actor.gather.remote() for actor in (self.ray_actors + [self.server])
                ]
                ray.get(all_ret)

                ray.get(self.server.global_update.remote())

                if global_rounds % validate_interval == 0 and global_rounds > 0:
                    ray.get(
                        [
                            actor.broadcast.remote()
                            for actor in self.ray_actors + [self.server]
                        ]
                    )
                    ret_ids = [
                        actor.evaluate.remote(
                            round_number=global_rounds,
                            metrics={"top1": top1_accuracy},
                        )
                        for actor in self.ray_actors
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
        wandb.log(r)
        # self.json_logger.info(r)
        return loss, top1
