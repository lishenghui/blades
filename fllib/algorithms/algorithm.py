from typing import Callable, Dict, Optional, Union

from ray.air.integrations.wandb import setup_wandb
from ray.rllib.utils import force_list
from ray.rllib.utils.from_config import from_config
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.logger import Logger
from ray.tune.resources import Resources
from ray.tune.trainable import Trainable
from ray.util.annotations import PublicAPI

from fllib.algorithms.algorithm_config import AlgorithmConfig
from fllib.algorithms.callbacks import AlgorithmCallbackList
from fllib.algorithms.client_manager import ClientManager
from fllib.types import ResultDict, PartialAlgorithmConfigDict


class Algorithm(Trainable):
    """Base class for all FL Algorithms."""

    client_manager_cls = ClientManager

    def __init__(
        self,
        config: Optional[AlgorithmConfig] = None,
        logger_creator: Optional[Callable[[], Logger]] = None,
        **kwargs,
    ):
        config = config or self.get_default_config()

        # Translate possible dict into an AlgorithmConfig object, as well as,
        # resolving generic config objects into specific ones (e.g. passing
        # an `AlgorithmConfig` super-class instance into a PPO constructor,
        # which normally would expect a PPOConfig object).
        if isinstance(config, dict):
            default_config = self.get_default_config()
            # `self.get_default_config()` also returned a dict ->
            # Last resort: Create core AlgorithmConfig from merged dicts.
            if isinstance(default_config, dict):
                config = AlgorithmConfig.from_dict(
                    config_dict=self.merge_trainer_configs(default_config, config, True)
                )
            # Default config is an AlgorithmConfig -> update its properties
            # from the given config dict.
            else:
                config = default_config.update_from_dict(config)
        else:
            default_config = self.get_default_config()
            # Given AlgorithmConfig is not of the same type as the default config:
            # This could be the case e.g. if the user is building an algo from a
            # generic AlgorithmConfig() object.
            if not isinstance(config, type(default_config)):
                config = default_config.update_from_dict(config.to_dict())

        # In case this algo is using a generic config (with no algo_class set), set it
        # here.
        if config.algo_class is None:
            config.algo_class = type(self)

        # Validate and freeze our AlgorithmConfig object (no more changes possible).
        config.validate()
        config.freeze()

        super().__init__(
            config=config,
            logger_creator=logger_creator,
            **kwargs,
        )

        # Check, whether `training_iteration` is still a tune.Trainable property
        # and has not been overridden by the user in the attempt to implement the
        # algos logic (this should be done now inside `training_step`).
        try:
            assert isinstance(self.training_iteration, int)
        except AssertionError as err:
            raise AssertionError(
                "Your Algorithm's `training_iteration` seems to be overridden by your "
                "custom training logic! To solve this problem, simply rename your "
                "`self.training_iteration()` method into `self.training_step`."
            ) from err

    def setup(self, config: AlgorithmConfig):
        # Setup our config: Merge the user-supplied config dict (which could
        # be a partial config dict) with the class' default.

        callback = from_config(config.pop("callbacks_config"))
        self.callbacks = AlgorithmCallbackList(force_list(callback))
        self.callbacks.setup(self)

        if not isinstance(config, AlgorithmConfig):
            assert isinstance(config, PartialAlgorithmConfigDict)
            config_obj = self.get_default_config()
            if not isinstance(config_obj, AlgorithmConfig):
                assert isinstance(config, PartialAlgorithmConfigDict)
                config_obj = AlgorithmConfig().from_dict(config_obj)
            config_obj.update_from_dict(config)
            self.config = config_obj
        if config.get("wandb_api_key", False):
            self.wandb = setup_wandb(
                config.to_dict(),
                # api_key_file="~/.wandb/api_key",
                api_key=config.get("wandb_api_key", None),
                trial_id=self.trial_id,
                trial_name=self.trial_name,
                group=config.get("wandb_group", None),
                project=config.get("wandb_project", None),
            )

    def step(self) -> ResultDict:
        # `self.iteration` gets incremented after this function returns,
        # meaning that e. g. the first time this function is called,
        # self.iteration will be 0.
        evaluate_this_iter = (
            self.config.evaluation_interval is not None
            and (self.iteration + 1) % self.config.evaluation_interval == 0
        )

        # Results dict for training (and if appolicable: evaluation).
        results: ResultDict = {}
        self.callbacks.on_train_round_begin()
        results = self.training_step()
        self.callbacks.on_train_round_end()
        if evaluate_this_iter:
            results.update(self.evaluate())

        if self.config.get("wandb_api_key", False):
            self.wandb.log(results)
        return results

    def training_step(self):
        """Performs a single training step and returns a dict of results."""
        return {}

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        """Returns the default config (AlgorithmConfig object)"""
        return AlgorithmConfig()

    # @OverrideToImplementCustomLogic
    @classmethod
    # @override(Trainable)
    def default_resource_request(
        cls, config: PartialAlgorithmConfigDict
    ) -> Union[Resources, PlacementGroupFactory]:
        # Default logic for RLlib Algorithms:
        # Create one bundle per individual worker (local or remote).
        # Use `num_cpus_for_driver` and `num_gpus` for the local worker and
        # `num_cpus_per_worker` and `num_gpus_per_worker` for the remote
        # workers to determine their CPU/GPU resource needs.

        # Convenience config handles.
        cf = dict(cls.get_default_config(), **config)
        # eval_cf = cf["evaluation_config"]

        local_worker = {
            "CPU": cf["num_cpus_for_driver"],
            "GPU": 0 if cf["_fake_gpus"] else cf["num_gpus_for_driver"],
        }
        remote_workers = [
            {
                "CPU": cf["num_cpus_per_worker"],
                "GPU": cf["num_gpus_per_worker"],
                **cf["custom_resources_per_worker"],
            }
            for _ in range(cf["num_remote_workers"])
        ]

        bundles = [local_worker] + remote_workers

        # In case our I/O reader/writer requires conmpute resources.
        # bundles += get_offline_io_resource_bundles(cf)

        # Return PlacementGroupFactory containing all needed resources
        # (already properly defined as device bundles).
        return PlacementGroupFactory(
            bundles=bundles,
            strategy=config.get("placement_strategy", "PACK"),
        )

    def evaluate(self):
        """Performs a single evaluation step and returns a dict of results."""

    @PublicAPI
    def __getstate__(self) -> Dict:
        """Returns current state of Algorithm, sufficient to restore it from
        scratch.

        Returns:
            The current state dict of this Algorithm, which can be used to sufficiently
            restore the algorithm from scratch without any other information.
        """
        # Add config to state so complete Algorithm can be reproduced w/o it.
        state = {
            "config": self.config,
        }

        return state
