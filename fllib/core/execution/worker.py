from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, TypeVar

import ray
from torch import Tensor

from fllib.constants import MAIN_ACTOR
from fllib.core.execution.scaling_config import WorkerGroupScalingConfig
from fllib.tasks import TaskSpec, Task
from fllib.types import ModelWeights
from fllib.utils.annotations import DeveloperAPI

T = TypeVar("T")


@DeveloperAPI
class Worker:
    def __init__(
        self,
        task_spec: TaskSpec = TaskSpec(),
        global_config: Dict[str, Any] = {},
        worker_scaling_config: WorkerGroupScalingConfig = WorkerGroupScalingConfig(),
        device: str = "cpu",
    ) -> None:
        # pick the configs that we need for the learner from scaling config
        # self._distributed = worker_scaling_config.get("num_workers", 0) > 1
        self._distributed = worker_scaling_config.num_workers > 1
        if worker_scaling_config.num_gpus_per_worker <= 0:
            self.device = "cpu"
        elif not self._distributed:
            self.device = "cuda"
        else:
            self.device = (
                "cuda" if worker_scaling_config.num_gpus_per_worker > 0 else device
            )
        # if we are using gpu but we are not distributed, use this gpu for training
        # self._local_gpu_idx = worker_scaling_config.local_gpu_idx
        global_worker = ray._private.worker.global_worker
        global_worker.actors[MAIN_ACTOR] = self

        self._states: Dict[str, Optional[Tensor]] = {}
        self._task_spec = task_spec
        self._task = None
        self._global_config = global_config
        self._current_client_id = None
        self._client_states = {}

    @property
    def global_config(self) -> Dict[str, Any]:
        """Global config."""
        return self._global_config

    @property
    def distributed(self) -> bool:
        """Whether the worker is running in distributed mode."""
        return self._distributed

    def on_global_var_update(self, global_vars):
        pass

    @property
    def task(self) -> Task:
        """Task instance."""
        return self._task

    def switch_client(self, client_id):
        if self._current_client_id is not None:
            self._client_states[
                self._current_client_id
            ] = self.task.get_optimizer_states()
        self._current_client_id = client_id
        opt_states = self._client_states.get(client_id, None)
        if opt_states is not None:
            self.task.set_optimizer_states(opt_states)

    def setup(self):
        """Setups the Worker.

        This method should be called before the learner is used. It is responsible for
        setting up the module and optimizers.
        """
        if self._task_spec:
            self._task = self._task_spec.build(self.device)

    def apply(
        self,
        func: Callable[["Worker", Optional[Any], Optional[Any]], T],
        *args,
        **kwargs,
    ) -> T:
        """Calls the given function with this worker instance.

        Useful for when the RolloutWorker class has been converted into a
        ActorHandle and the user needs to execute some functionality (e.g.
        add a property) on the underlying policy object.

        Args:
            func: The function to call, with this Worker as first
                argument, followed by args, and kwargs.
            args: Optional additional args to pass to the function call.
            kwargs: Optional additional kwargs to pass to the function call.

        Returns:
            The return value of the function call.
        """
        return func(self, *args, **kwargs)

    def get_state(self, name: str) -> ModelWeights:
        return self._states[name]

    def set_state(self, name: str, tensor: Optional[Tensor]) -> None:
        self._states[name] = tensor


@dataclass
class WorkerSpec:
    worker_class: Type["Worker"]
    task_spec: Dict[str, Any] = field(default_factory=dict)
    global_config: Dict[str, Any] = field(default_factory=dict)
    worker_scaling_config: Dict = field(default_factory=dict)

    def get_params_dict(self) -> Dict[str, Any]:
        """Returns the parameters than be passed to the Learner constructor."""
        return {
            "task_spec": self.task_spec,
            "global_config": self.global_config,
            "worker_scaling_config": self.worker_scaling_config,
        }

    def build(self) -> "Worker":
        """Builds the Learner instance."""
        return self.worker_class(**self.get_params_dict())
