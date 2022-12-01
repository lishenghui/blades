import logging
import sys
from typing import Optional

import numpy as np
import ray
import wandb
from tqdm import trange

from blades.core.communicator import assign_rank
from blades.core.worker import Worker
from blades.datasets.fldataset import FLDataset
from blades.utils.utils import top1_accuracy
from .config import ScalingConfig, RunConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Simulator(object):
    def __init__(
        self,
        dataset: FLDataset,
        scaling_config: Optional[ScalingConfig] = None,
        run_config: Optional[RunConfig] = None,
    ) -> None:

        self.scaling_config = (
            scaling_config if scaling_config is not None else ScalingConfig()
        )
        self.run_config = run_config if run_config is not None else RunConfig()
        num_actors = self.scaling_config.num_workers

        self.clients = run_config.clients
        client_groups = np.array_split(self.clients, num_actors)

        server_kws = run_config.server_kws
        server_kws["clients"] = self.clients

        server_kws |= {
            "model": run_config.global_model,
            "clients": self.clients,
        }
        self.server = run_config.server_cls.options(
            num_gpus=scaling_config.num_gpus_server
        ).remote(**server_kws)

        ray.get(self.server.init.remote())

        self.ray_actors = [
            Worker.options(num_gpus=scaling_config.num_gpus_per_worker).remote(
                dataset,
                run_config.global_model,
                run_config.local_opt_cls,
                run_config.local_opt_kws,
                clients=client_groups[i],
            )
            for i in range(num_actors)
        ]
        ray.get([actor.init.remote() for actor in self.ray_actors])

        assign_rank(self.server, self.ray_actors)

    def run(self):
        global_rounds = self.run_config.global_steps
        validate_interval = self.run_config.validate_interval
        local_steps = self.run_config.local_steps

        with trange(0, global_rounds + 1) as t:
            for global_rounds in t:
                ray.get(
                    [
                        actor.broadcast.remote()
                        for actor in self.ray_actors + [self.server]
                    ]
                )

                ray.get(
                    [
                        actor.local_train.remote(num_steps=local_steps)
                        for actor in self.ray_actors
                    ]
                )

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
        return loss, top1
