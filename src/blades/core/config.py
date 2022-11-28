from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    Tuple,
    Any,
)

from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.tune.callback import Callback
    from ray.tune.progress_reporter import ProgressReporter
    from ray.tune.stopper import Stopper
    from ray.tune.syncer import SyncConfig
    from ray.tune.utils.log import Verbosity

# Adopted from from
# https://github.com/ray-project/ray/blob/master/python/ray/air/config.py


MAX = "max"
MIN = "min"


@dataclass
@PublicAPI(stability="beta")
class ScalingConfig:
    """Configuration for scaling training.

    Args:
        server_resources: Resources to allocate for the trainer. If None is provided,
            will default to 1 CPU.
        num_workers: The number of workers (Ray actors) to launch.
            Each worker will reserve 1 CPU by default. The number of CPUs
            reserved by each worker can be overridden with the
            ``resources_per_worker`` argument.
        use_gpu: If True, training will be done on GPUs (1 per worker).
            Defaults to False. The number of GPUs reserved by each
            worker can be overridden with the ``resources_per_worker``
            argument.
        resources_per_worker: If specified, the resources
            defined in this Dict will be reserved for each worker. The
            ``CPU`` and ``GPU`` keys (case-sensitive) can be defined to
            override the number of CPU/GPUs used by each worker.
        placement_strategy: The placement strategy to use for the
            placement group of the Ray actors. See :ref:`Placement Group
            Strategies <pgroup-strategy>` for the possible options.
        _max_cpu_fraction_per_node: (Experimental) The max fraction of CPUs per node
            that Train will use for scheduling training actors. The remaining CPUs
            can be used for dataset tasks. It is highly recommended that you set this
            to less than 1.0 (e.g., 0.8) when passing datasets to trainers, to avoid
            hangs / CPU starvation of dataset tasks. Warning: this feature is
            experimental and is not recommended for use with autoscaling (scale-up will
            not trigger properly).
    """

    # If adding new attributes here, please also update
    # ray.train.gbdt_trainer._convert_scaling_config_to_ray_params
    server_resources: Optional[Dict] = None
    num_workers: Optional[int] = None
    use_gpu: bool = False
    resources_per_worker: Optional[Dict] = None

    def __post_init__(self):
        self.use_gpu = self.num_gpus_per_worker > 0
        if self.resources_per_worker:

            if not self.use_gpu and self.num_gpus_per_worker > 0:
                raise ValueError(
                    "`use_gpu` is False but `GPU` was found in "
                    "`resources_per_worker`. Either set `use_gpu` to True or "
                    "remove `GPU` from `resources_per_worker."
                )

            if self.use_gpu and self.num_gpus_per_worker == 0:
                raise ValueError(
                    "`use_gpu` is True but `GPU` is set to 0 in "
                    "`resources_per_worker`. Either set `use_gpu` to False or "
                    "request a positive number of `GPU` in "
                    "`resources_per_worker."
                )

    @property
    def num_gpus_per_worker(self):
        """The number of GPUs to set per worker."""
        return self._resources_per_worker_not_none.get("GPU", 0)

    @property
    def num_gpus_server(self):
        """The number of GPUs to set per worker."""
        return self._server_resources_not_none.get("GPU", 0)

    @property
    def _resources_per_worker_not_none(self):
        if self.resources_per_worker is None:
            if self.use_gpu:
                # Note that we don't request any CPUs, which avoids possible
                # scheduling contention. Generally nodes have many more CPUs than
                # GPUs, so not requesting a CPU does not lead to oversubscription.
                return {"GPU": 1}
            else:
                return {"CPU": 1}
        resources_per_worker = {
            k: v for k, v in self.resources_per_worker.items() if v != 0
        }
        resources_per_worker.setdefault("GPU", int(self.use_gpu))
        return resources_per_worker

    @property
    def _server_resources_not_none(self):
        if self.server_resources is None:
            if self.num_workers:
                # For Google Colab, don't allocate resources to the base Trainer.
                # Colab only has 2 CPUs, and because of this resource scarcity,
                # we have to be careful on where we allocate resources. Since Colab
                # is not distributed, the concern about many parallel Ray Tune trials
                # leading to all Trainers being scheduled on the head node if we set
                # `trainer_resources` to 0 is no longer applicable.
                try:
                    import google.colab  # noqa: F401

                    trainer_resources = 0
                except ImportError:
                    trainer_resources = 1
            else:
                # If there are no additional workers, then always reserve 1 CPU for
                # the Trainer.
                trainer_resources = 1

            return {"CPU": trainer_resources}
        return {k: v for k, v in self.server_resources.items() if v != 0}


@dataclass
@PublicAPI(stability="beta")
class RunConfig:
    global_model: Optional[str] = None
    validate_interval: Union[int, "Verbosity"] = 3
    local_steps: Union[int, "Verbosity"] = 3
    global_steps: Union[int, "Verbosity"] = 3
    log_to_file: Union[bool, str, Tuple[str, str]] = False

    server_cls: Optional[Union[type]] = None
    server_kws: Optional[Union[Dict]] = None

    clients: List[Any] = None
    local_opt_cls: Optional[type] = None
    local_opt_kws: Optional[Union[Dict]] = None


@dataclass
@PublicAPI(stability="beta")
class ClientConfig:
    name: Optional[str] = None
    local_dir: Optional[str] = None
    callbacks: Optional[List["Callback"]] = None
    stop: Optional[Union[Mapping, "Stopper", Callable[[str, Mapping], bool]]] = None
    sync_config: Optional["SyncConfig"] = None
    progress_reporter: Optional["ProgressReporter"] = None
    verbose: Union[int, "Verbosity"] = 3
    log_to_file: Union[bool, str, Tuple[str, str]] = False
