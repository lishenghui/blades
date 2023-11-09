from typing import Type, Optional, Union

from ray.rllib.utils.from_config import NotProvided

from fllib.core.execution.scaling_config import WorkerGroupScalingConfig
from fllib.core.execution.worker import Worker
from fllib.core.execution.worker_group import WorkerGroup
from fllib.tasks import TaskSpec


# if TYPE_CHECKING:
#     from fllib.core.execution.worker import Worker


class WorkerGroupConfig:
    """Configuration object for WorkerGroup."""

    def __init__(self, cls: Type[WorkerGroup] = None, algo_config={}) -> None:
        # Define the default WorkerGroup class
        self.worker_group_class = cls or WorkerGroup

        # `self.task()`
        self.task_spec = None
        # `self.worker()`
        self.worker_class = Worker

        # `self.resources()`
        self.num_gpus_per_worker = 0
        self.num_cpus_per_worker = 1
        self.num_remote_workers = 1

        self.algo_config = algo_config

        # TODO (Avnishn): We should come back and revise how to specify algorithm
        # resources this is a stop gap solution for now so that users can specify the
        # local gpu id to use when training with gpu and local mode. I doubt this will
        # be used much since users who have multiple gpus will probably be fine with
        # using the 0th gpu or will use multi gpu training.
        self.local_gpu_idx = 0

    def validate(self) -> None:
        if self.worker_class is None:
            raise ValueError(
                "Cannot initialize an Learner without an Worker class. Please provide "
                "the Worker class with .worker(worker_class=MyWorkerClass)."
            )

        # if self.task_spec is None:
        #     raise ValueError(
        #         "Cannot initialize an Learner without an Worker class. Please provide"
        #         "the Worker class with .worker(worker_class=MyWorkerClass)."
        #     )

    def task(self, task_spec: TaskSpec = NotProvided) -> "WorkerGroupConfig":
        if task_spec is not NotProvided:
            self.task_spec = task_spec
        return self

    def build(self) -> WorkerGroup:
        from fllib.core.execution.worker import WorkerSpec

        self.validate()

        scaling_config = WorkerGroupScalingConfig(
            num_workers=self.num_remote_workers,
            num_gpus_per_worker=self.num_gpus_per_worker,
            num_cpus_per_worker=self.num_cpus_per_worker,
            local_gpu_idx=self.local_gpu_idx,
        )

        worker_spec = WorkerSpec(
            worker_class=self.worker_class,
            task_spec=self.task_spec,
            worker_scaling_config=scaling_config,
            global_config=self.algo_config,
        )
        return self.worker_group_class(worker_spec)

    def resources(
        self,
        num_remote_workers: Optional[int] = NotProvided,
        num_gpus_per_worker: Optional[Union[float, int]] = NotProvided,
        num_cpus_per_worker: Optional[Union[float, int]] = NotProvided,
        local_gpu_idx: Optional[int] = NotProvided,
    ) -> "WorkerGroupConfig":
        if num_remote_workers is not NotProvided:
            self.num_remote_workers = num_remote_workers
        if num_gpus_per_worker is not NotProvided:
            self.num_gpus_per_worker = num_gpus_per_worker
        if num_cpus_per_worker is not NotProvided:
            self.num_cpus_per_worker = num_cpus_per_worker
        if local_gpu_idx is not NotProvided:
            self.local_gpu_idx = local_gpu_idx

        return self

    def worker(
        self,
        *,
        worker_class: Optional[Type["Worker"]] = NotProvided,
    ) -> "WorkerGroupConfig":
        if worker_class is not NotProvided:
            self.worker_class = worker_class

        return self
