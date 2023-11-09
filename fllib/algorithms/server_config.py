import copy
import logging
from typing import Optional, Callable, Dict  # , Tuple, List

# from ray.tune.logger import Logger
from ray.rllib.utils.from_config import from_config
from ray.util import log_once

# from fllib.clients.callbacks import ClientCallback
from fllib.types import NotProvided, PartialAlgorithmConfigDict
from fllib.types import TYPE_CHECKING

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class ServerConfig:
    def __init__(self, class_specifier, task_spec) -> None:
        super().__init__()

        self.class_specifier = class_specifier
        self.task_spec = task_spec

        # `self.training()`
        self.aggregator = "Mean"
        self.optimizer = {}

    def training(
        self,
        task_spec: Optional[Callable] = NotProvided,
        aggregator: Optional[Dict] = NotProvided,
        optimizer: Optional[Dict] = NotProvided,
    ) -> "ServerConfig":
        if task_spec is not NotProvided:
            self.task_spec = task_spec
        if aggregator is not NotProvided:
            self.aggregator = aggregator
        if optimizer is not NotProvided:
            self.optimizer = optimizer
        return self

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
        return getattr(self, item)

    def build(self, use_copy: bool = True, device: str = "cpu") -> Callable:
        """Build the client.

        Args:
            use_copy (bool, optional): Defaults to True.

        Returns:
            Callable: the instantiated client.
        """
        server_cls = self.class_specifier
        if server_cls is None:
            raise ValueError("server_cls is not set")
        return from_config(
            server_cls,
            None,
            task_spec=self.task_spec,
            device=device,
            optimizer_config=self.optimizer,
            aggregator_config=self.aggregator,
        )

    def to_dict(self) -> dict:
        """Converts all settings into a legacy config dict for backward
        compatibility.

        Returns:
            A complete AlgorithmConfigDict, usable in backward-compatible Tune/RLlib
            use cases, e.g. w/ `tune.Tuner().fit()`.
        """
        config = copy.deepcopy(vars(self))

        return config

    def update_from_dict(
        self,
        config_dict: PartialAlgorithmConfigDict,
    ) -> "ServerConfig":
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

        # Modify our properties one by one.
        for key, value in config_dict.items():
            # Some keys specify config sub-dicts and therefore should go through the
            # correct methods to properly `.update()` those from given config dict
            # (to not lose any sub-keys).

            if key == "lr":
                self.optimizer.update(lr=value)
            if key in ["aggregator", "optimizer"]:
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
        return self
