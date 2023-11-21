import copy
import logging
from typing import Callable, Optional, Type, Union

from ray.tune.logger import Logger
from ray.tune.registry import _global_registry
from ray.tune.registry import get_trainable_cls

# from ray.tune.result import TRIAL_INFO
from ray.util import log_once

from fllib.algorithms.callbacks import AlgorithmCallback
from fllib.algorithms.server_config import ServerConfig
from fllib.clients import ClientConfig
from fllib.constants import FLLIB_DATASET
from fllib.core.execution.worker import Worker
from fllib.core.execution.worker_group_config import WorkerGroupConfig
from fllib.tasks import TaskSpec
from fllib.types import (
    AlgorithmConfigDict,
    PartialAlgorithmConfigDict,
    TYPE_CHECKING,
    NotProvided,
)

if TYPE_CHECKING:
    from fllib.algorithms.algorithm import Algorithm

logger = logging.getLogger(__name__)


class AlgorithmConfig:
    """A RLlib AlgorithmConfig builds an RLlib Algorithm from a given
    configuration.

    Example:
        >>> from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
        >>> # Construct a generic config object, specifying values within different
        >>> # sub-categories, e.g. "training".
        >>> config = AlgorithmConfig().training(gamma=0.9, lr=0.01)
        ...              .environment(env="CartPole-v1")
        ...              .resources(num_gpus=0)
        ...              .rollouts(num_rollout_workers=4)
        ...              .callbacks(MemoryTrackingCallbacks)
        >>> # A config object can be used to construct the respective Trainer.
        >>> rllib_trainer = config.build()

    Example:
        >>> from ray import tune
        >>> # In combination with a tune.grid_search:
        >>> config = AlgorithmConfig()
        >>> config.training(lr=tune.grid_search([0.01, 0.001]))
        >>> # Use `to_dict()` method to get the legacy plain Python config dict
        >>> # for usage with `tune.Tuner().fit()`.
        >>> tune.Tuner("[registered trainer class]", param_space=config.to_dict()).fit()
    """

    def __init__(self, algo_class=None):
        # Define all settings and their default values.

        self.random_seed = 1234

        # self.training()
        self.algo_class = algo_class
        self.global_model = None
        self.num_batch_per_round = 1
        self.client_config = {}
        self.server_config = {}
        self.logger_creator = None
        self.learner_class = None
        self.task_config = {"task_class": "fllib.tasks.GeneralClassification"}

        # experimental: this will contain the hyper-parameters that are passed to the
        # Learner, for computing loss, etc. New algorithms have to set this to their
        # own default. .training() will modify the fields of this object.
        # self._learner_hps = LearnerHPs()
        # Has this config object been frozen (cannot alter its attributes anymore).
        self._is_frozen = False

        # `self.resources()`
        self.num_cpus_for_driver = 1
        self.num_gpus_for_driver = 0
        self.num_cpus_per_worker = 1
        self.num_gpus_per_worker = 0
        self.num_remote_workers = 1
        self.num_gpus = 0
        self._fake_gpus = False
        self.custom_resources_per_worker = {}
        self.placement_strategy = "PACK"

        # self.data()`
        self.dataset_config = {}
        self.splitter_config = {}
        self.num_clients = 0

        self.reuse_actors = True

        # `self.evaluation()`
        self.evaluation_interval = 10
        self.evaluation_config = None

        self.callbacks_config = AlgorithmCallback

    def client(self, *, client_config: Optional[dict] = NotProvided):
        if client_config is not NotProvided:
            self.client_config = client_config
        return self

    def data(
        self,
        *,
        dataset: Optional[str] = NotProvided,
        num_clients: Optional[Union[float, int]] = NotProvided,
        dataset_config=NotProvided,
    ):
        if dataset is not NotProvided:
            self.dataset = dataset
        if num_clients is not NotProvided:
            self.num_clients = num_clients
        if dataset_config is not NotProvided:
            self.dataset_config = dataset_config
        return self

    def training(
        self,
        *,
        global_model: Optional[str] = NotProvided,
        local_lr: Optional[float] = NotProvided,
        num_batch_per_round: Optional[int] = NotProvided,
        server_config: Optional[dict] = NotProvided,
    ):
        if local_lr is not NotProvided:
            self.local_lr = local_lr
        if num_batch_per_round is not NotProvided:
            self.num_batch_per_round = num_batch_per_round
        if global_model is not NotProvided:
            self.global_model = global_model
        if server_config is not NotProvided:
            self.server_config = server_config
        return self

    def resources(
        self,
        *,
        num_cpus_per_worker: Optional[Union[float, int]] = NotProvided,
        num_gpus_per_worker: Optional[Union[float, int]] = NotProvided,
        num_cpus_for_driver: Optional[Union[float, int]] = NotProvided,
        num_gpus_for_driver: Optional[Union[float, int]] = NotProvided,
        num_remote_workers: Optional[Union[float, int]] = NotProvided,
    ):
        if num_cpus_per_worker is not NotProvided:
            self.num_cpus_per_worker = num_cpus_per_worker
        if num_gpus_per_worker is not NotProvided:
            self.num_gpus_per_worker = num_gpus_per_worker
        if num_cpus_for_driver is not NotProvided:
            self.num_cpus_for_driver = num_cpus_for_driver
        if num_gpus_for_driver is not NotProvided:
            self.num_gpus_for_driver = num_gpus_for_driver
        if num_remote_workers is not NotProvided:
            self.num_remote_workers = num_remote_workers
        return self

    def get_task_spec(self) -> TaskSpec:
        return TaskSpec(task_class=self.task_config["task_class"], alg_config=self)

    def get_client_config(self) -> ClientConfig:
        if not self._is_frozen:
            raise ValueError(
                "Cannot call `get_client_config()` on an unfrozen "
                "AlgorithmConfig! Please call `freeze()` first."
            )

        config = (
            ClientConfig(class_specifier="fllib.clients.Client")
            .training(
                num_batch_per_round=self.num_batch_per_round,
                lr=self.local_lr,
            )
            .update_from_dict(self.client_config)
        )

        return config

    def get_server_config(self) -> ServerConfig:
        if not self._is_frozen:
            raise ValueError(
                "Cannot call `get_server_config()` on an unfrozen "
                "AlgorithmConfig! Please call `freeze()` first."
            )

        config = ServerConfig(
            class_specifier="fllib.algorithms.Server",
            task_spec=self.get_task_spec(),
        ).update_from_dict(self.server_config)

        return config

    def get_worker_group_config(self) -> WorkerGroupConfig:
        if not self._is_frozen:
            raise ValueError(
                "Cannot call `get_learner_group_config()` on an unfrozen "
                "AlgorithmConfig! Please call `freeze()` first."
            )

        config = (
            WorkerGroupConfig()
            .resources(
                num_remote_workers=self.num_remote_workers,
                num_cpus_per_worker=self.num_cpus_per_worker,
                num_gpus_per_worker=self.num_gpus_per_worker,
            )
            .worker(worker_class=Worker)
        )
        return config

    def get_default_worker_class(self) -> Union[Type["Worker"], str]:
        """Returns the Learner class to use for this algorithm.

        Override this method in the sub-class to return the Learner class type given
        the input framework.

        Returns:
            The Learner class to use for this algorithm either as a class type or as
            a string (e.g. ray.rllib.core.learner.testing.torch.BCTrainer).
        """
        raise NotImplementedError

    def build(
        self,
        logger_creator: Optional[Callable[[], Logger]] = None,
        use_copy: bool = True,
    ) -> "Algorithm":
        """Builds an Algorithm from the AlgorithmConfig.

        Args:
            env: Name of the environment to use (e.g. a gym-registered str),
                a full class path (e.g.
                "ray.rllib.examples.env.random_env.RandomEnv"), or an Env
                class directly. Note that this arg can also be specified via
                the "env" key in `config`.
            logger_creator: Callable that creates a ray.tune.Logger
                object. If unspecified, a default logger is created.

        Returns:
            A ray.rllib.algorithms.algorithm.Algorithm object.
        """
        if logger_creator is not None:
            self.logger_creator = logger_creator

        algo_class = self.algo_class
        if isinstance(self.algo_class, str):
            algo_class = get_trainable_cls(self.algo_class)

        return algo_class(
            config=self if not use_copy else copy.deepcopy(self),
            logger_creator=self.logger_creator,
        )

    def __getitem__(self, item):
        """Shim method to still support accessing properties by key lookup.

        This way, an AlgorithmConfig object can still be used as if a dict, e.g.
        by Ray Tune.

        Examples:
            >>> from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
            >>> config = AlgorithmConfig()
            >>> print(config["lr"])
            ... 0.001
        """
        # TODO: Uncomment this once all algorithms use AlgorithmConfigs under the
        #  hood (as well as Ray Tune).
        # if log_once("algo_config_getitem"):
        #    logger.warning(
        #        "AlgorithmConfig objects should NOT be used as dict! "
        #        f"Try accessing `{item}` directly as a property."
        #    )
        # In case user accesses "old" keys, which need to
        # be translated to their correct property names.
        item = self._translate_special_keys(item)
        return getattr(self, item)

    def __setitem__(self, key, value):
        # TODO: Remove comments once all methods/functions only support
        #  AlgorithmConfigs and there is no more ambiguity anywhere in the code
        #  on whether an AlgorithmConfig is used or an old python config dict.
        # raise AttributeError(
        #    "AlgorithmConfig objects should not have their values set like dicts"
        #    f"(`config['{key}'] = {value}`), "
        #    f"but via setting their properties directly (config.{prop} = {value})."
        # )
        if key == "multiagent":
            raise AttributeError(
                "Cannot set `multiagent` key in an AlgorithmConfig!\nTry setting "
                "the multi-agent components of your AlgorithmConfig object via the "
                "`multi_agent()` method and its arguments.\nE.g. `config.multi_agent("
                "policies=.., policy_mapping_fn.., policies_to_train=..)`."
            )
        super().__setattr__(key, value)

    def validate(self) -> None:
        """Validates all values in this config."""

        # Check that the `num_clients` is set correctly.
        if self.dataset_config.get("custom_dataset"):
            dataset_cls = _global_registry.get(
                FLLIB_DATASET, self.dataset_config["custom_dataset"]
            )
            self.num_clients = dataset_cls.num_clients
        else:
            if self.dataset_config.get("num_clients", None) is None:
                if self.num_clients < 1:
                    raise ValueError(
                        "You must specify a `num_clients` in your `dataset_config`",
                        "otherwise specify a `num_clients`!",
                    )
                self.dataset_config["num_clients"] = self.num_clients
            else:
                if self.num_clients > 0:
                    raise ValueError(
                        "You cannot specify both `num_clients` and `batch_size` ",
                        " in your `dataset_config`",
                    )
                self.num_clients = self.dataset_config["num_clients"]
            self.splitter_config["num_clients"] = self.num_clients
            if self.dataset_config.get("random_seed", None) is None:
                self.dataset_config["random_seed"] = self.random_seed
            self.splitter_config["random_seed"] = self.dataset_config["random_seed"]
            self.dataset_config["splitter_config"] = self.splitter_config
        if self.dataset_config.get("random_seed", None) is None:
            self.dataset_config["random_seed"] = self.random_seed

    def freeze(self) -> None:
        """Freezes this config object, such that no attributes can be set
        anymore.

        Algorithms should use this method to make sure that their config objects remain
        read-only after this.
        """
        if self._is_frozen:
            return
        self._is_frozen = True

        # Also freeze underlying eval config, if applicable.
        # if isinstance(self.evaluation_config, AlgorithmConfig):
        #     self.evaluation_config.freeze()

        # TODO: Flip out all set/dict/list values into frozen versions
        #  of themselves? This way, users won't even be able to alter those values
        #  directly anymore.

    def to_dict(self) -> AlgorithmConfigDict:
        """Converts all settings into a legacy config dict for backward
        compatibility.

        Returns:
            A complete AlgorithmConfigDict, usable in backward-compatible Tune/RLlib
            use cases, e.g. w/ `tune.Tuner().fit()`.
        """
        config = copy.deepcopy(vars(self))
        config.pop("algo_class")
        config.pop("_is_frozen", None)

        # Worst naming convention ever: NEVER EVER use reserved key-words...
        if "lambda_" in config:
            assert hasattr(self, "lambda_")
            config["lambda"] = getattr(self, "lambda_")
            config.pop("lambda_")
        if "input_" in config:
            assert hasattr(self, "input_")
            config["input"] = getattr(self, "input_")
            config.pop("input_")

        return config

    def get(self, key, default=None):
        """Shim method to help pretend we are a dict."""
        prop = self._translate_special_keys(key)
        return getattr(self, prop, default)

    def pop(self, key, default=None):
        """Shim method to help pretend we are a dict."""
        return self.get(key, default)

    def keys(self):
        """Shim method to help pretend we are a dict."""
        return self.to_dict().keys()

    def values(self):
        """Shim method to help pretend we are a dict."""
        return self.to_dict().values()

    def items(self):
        """Shim method to help pretend we are a dict."""
        return self.to_dict().items()

    @staticmethod
    def _translate_special_keys(key: str) -> str:
        # Handle special key (str) -> `AlgorithmConfig.[some_property]` cases.
        if key == "callbacks":
            key = "callbacks_config"
        elif key == "custom_eval_function":
            key = "custom_evaluation_function"
        elif key == "framework":
            key = "framework_str"
        elif key == "input":
            key = "input_"
        elif key == "lambda":
            key = "lambda_"

        return key

    def update_from_dict(
        self,
        config_dict: PartialAlgorithmConfigDict,
    ) -> "AlgorithmConfig":
        """Modifies this AlgorithmConfig via the provided python config dict.

        Warns if `config_dict` contains deprecated keys.
        Silently sets even properties of `self` that do NOT exist. This way, this method
        may be used to configure custom Policies which do not have their own specific
        AlgorithmConfig classes, e.g.
        `ray.rllib.examples.policy.random_policy::RandomPolicy`.

        Args:
            config_dict: The old-style python config dict (PartialAlgorithmConfigDict)
                to use for overriding some properties defined in there.

        Returns:
            This updated AlgorithmConfig object.
        """
        eval_call = {}

        # Modify our properties one by one.
        for key, value in config_dict.items():
            key = self._translate_special_keys(key)

            # Ray Tune saves additional data under this magic keyword.
            # This should not get treated as AlgorithmConfig field.
            # if key == TRIAL_INFO:
            #     continue

            # Some keys specify config sub-dicts and therefore should go through the
            # correct methods to properly `.update()` those from given config dict
            # (to not lose any sub-keys).
            # if key == "callbacks_config":
            # self.callbacks(callbacks_config=value)
            if key.startswith("evaluation_"):
                eval_call[key] = value
            elif key in ["model", "optimizer", "replay_buffer_config"]:
                self.training(**{key: value})
            # If config key matches a property, just set it, otherwise, warn and set.
            else:
                if not hasattr(self, key) and log_once(
                    "unknown_property_in_algo_config"
                ):
                    logger.warning(
                        f"Cannot create {type(self).__name__} from given "
                        f"`config_dict`! Property {key} not supported."
                    )
                setattr(self, key, value)

        self.evaluation(**eval_call)

        return self

    def evaluation(
        self,
        *,
        evaluation_interval: Optional[int] = NotProvided,
        evaluation_config: Optional[
            Union["AlgorithmConfig", PartialAlgorithmConfigDict]
        ] = NotProvided,
    ) -> "AlgorithmConfig":
        """Configures evaluation settings."""
        if evaluation_interval is not NotProvided:
            self.evaluation_interval = evaluation_interval
