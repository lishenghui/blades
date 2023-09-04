from typing import Dict, Optional, TypeVar

import ray
import torch
import torch.distributed as dist

# from ray.train._internal.worker_group import WorkerGroup
from fllib._internal.worker_group import WorkerGroup
from fllib.communication.communicator import Communicator
from fllib.constants import MINIMUM_GPU_FRACTION

# Generic type var for foreach_* methods.
T = TypeVar("T")


# communicator_set = WorkerGroup(
#     num_workers=4, num_cpus_per_worker=1, actor_cls=Communicator, eager_install=False
# )


class CommunicationSet(WorkerGroup):
    def __init__(
        self,
        num_workers: int = 1,
        num_cpus_per_worker: float = 1,
        num_gpus_per_worker: float = MINIMUM_GPU_FRACTION,
        additional_resources_per_worker: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            num_workers,
            num_cpus_per_worker,
            num_gpus_per_worker,
            additional_resources_per_worker,
            actor_cls=Communicator,
            placement_group=None,
            eager_install=False,
        )
        results = []
        for i in range(num_workers):
            device = torch.device(f"cuda:{i}")
            res = self.workers[i].actor.setup.remote(
                world_size=num_workers,
                world_rank=i,
                cuda=i % 4,
                cuda_visible_devices="0,1,2,3",
            )
            self.workers[i].metadata.device = device
            results.append(res)
        ray.get(results)

    def _worker_by_device(self, device: int):
        for worker in self.workers:
            if device == worker.metadata.device:
                return worker
        raise ValueError(f"Could not find worker for device {device}")

    def broadcast(
        self,
        tensor: torch.Tensor,
    ):
        # print("shape" * 10, tensor.shape)
        sender = self._worker_by_device(tensor.device)
        receivers = [w for w in self.workers if w != sender]

        if not hasattr(sender, "queue"):
            sender.queue = ray.get(sender.actor.get_msg_queue.remote())

        sender.queue.put(tensor)

        def func_recive(worker):
            if not hasattr(worker, "data"):
                worker.data = torch.ones_like(tensor).to(worker.device)
            # dist.all_reduce(worker.data)
            return
            dist.broadcast(worker.data, src=0)

        def func_send(worker):
            #     worker.data = torch.ones(464154).to("cuda:0")
            # worker.data = tensor
            # dist.broadcast(worker.data, src=0)
            tensor = sender.queue.get()
            # dist.all_reduce(tensor)
            return
            dist.broadcast(tensor, src=0)

        sender_rerult = sender.actor.apply.remote(func_send)
        rec_rerults = [w.actor.apply.remote(func_recive) for w in receivers]
        ray.get([sender_rerult] + rec_rerults)
