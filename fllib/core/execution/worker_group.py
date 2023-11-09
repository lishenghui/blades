from dataclasses import dataclass
from typing import Any, Callable, List, TypeVar, Union

import ray
from ray.actor import ActorHandle
from ray.rllib.utils.actor_manager import RemoteCallResults
from ray.train._internal.backend_executor import BackendExecutor

from fllib.core.execution.actor_manager import ActorManager

# Generic type var for foreach_* methods.
T = TypeVar("T")


# Disable sharing CUDA_VISIBLE_DEVICES for all workers.
def _get_backend_config(worker_class) -> str:
    _ = worker_class
    from ray.train.torch import TorchConfig

    @dataclass
    class BackendConfig(TorchConfig):
        @property
        def backend_cls(self):
            cls = super().backend_cls
            cls.share_cuda_visible_devices = False
            return cls

    return BackendConfig()


class WorkerGroup:
    def __init__(self, remote_worker_spec):
        scaling_config = remote_worker_spec.worker_scaling_config
        worker_class = remote_worker_spec.worker_class
        self._workers = None
        self._local_worker = None

        if scaling_config.num_workers:
            backend_config = _get_backend_config(worker_class)
            backend_executor = BackendExecutor(
                backend_config=backend_config,
                num_workers=scaling_config.num_workers,
                num_cpus_per_worker=scaling_config.num_cpus_per_worker,
                num_gpus_per_worker=scaling_config.num_gpus_per_worker,
                max_retries=0,
            )
            backend_executor.start(
                train_cls=worker_class,
                train_cls_kwargs=remote_worker_spec.get_params_dict(),
            )
            self._backend_executor = backend_executor

            self._workers = [w.actor for w in backend_executor.worker_group.workers]
            # run the neural network building code on remote workers
            ray.get([w.setup.remote() for w in self._workers])
            # use only 1 max in flight request per worker since training workers have to
            # be synchronously executed.
            self._worker_manager = ActorManager(
                self._workers,
                max_remote_requests_in_flight_per_actor=1,
            )
        else:
            self._local_worker = worker_class(**remote_worker_spec.get_params_dict())
            self._local_worker.setup()

    @property
    def workers(self) -> List[ActorHandle]:
        return self._workers

    def local_worker(self) -> Any:
        return self._local_worker

    def sync_state(self, name: str, source_state: Any) -> None:
        """Sync global weights to given WorkerSet or list of workers."""
        if self._workers:
            results = self._worker_manager.foreach_actor(
                lambda w: w.task.set_weights(source_state)
                # lambda w: w.set_state(name, source_state)
            )
        else:
            results = self.local_worker().task.set_weights(source_state)
        return results

    def foreach_actor(
        self,
        func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        *,
        healthy_only=True,
        remote_actor_ids: List[int] = None,
        timeout_seconds=None,
        return_obj_refs: bool = False,
        mark_healthy: bool = False,
    ) -> RemoteCallResults:
        """Calls the given function with each actor instance as arg.

        Automatically mark actors unhealthy if they fail to respond.

        Args:
            func: A single, or a list of Callables, that get applied on the list
                of specified remote actors.
            healthy_only: If True, applies func on known healthy actors only.
            remote_actor_ids: Apply func on a selected set of remote actors.
            timeout_seconds: Ray.get() timeout. Default is None.
                Note(jungong) : setting timeout_seconds to 0 effectively makes all the
                remote calls fire-and-forget, while setting timeout_seconds to None
                make them synchronous calls.
            return_obj_refs: whether to return ObjectRef instead of actual results.
                Note, for fault tolerance reasons, these returned ObjectRefs should
                never be resolved with ray.get() outside of the context of this manager.
            mark_healthy: whether to mark certain actors healthy based on the results
                of these remote calls. Useful, for example, to make sure actors
                do not come back without proper state restoration.

        Returns:
            The list of return values of all calls to `func(actor)`. The values may be
            actual data returned or exceptions raised during the remote call in the
            format of RemoteCallResults.
        """
        return self._worker_manager.foreach_actor(
            func=func,
            healthy_only=healthy_only,
            remote_actor_ids=remote_actor_ids,
            timeout_seconds=timeout_seconds,
            return_obj_refs=return_obj_refs,
            mark_healthy=mark_healthy,
        )

    def execute_with_actor_pool(
        self,
        func: Callable,
        values: List[Any],
        affinity_actors: List[List[ActorHandle]] = None,
    ):
        return self._worker_manager.execute_with_actor_pool(
            func=func, values=values, affinity_actors=affinity_actors
        )

    def local_execute(
        self,
        func: Callable,
        values: List[Any],
    ):
        results = []
        for value in values:
            results.append(func(self.local_worker(), value))
        return results
