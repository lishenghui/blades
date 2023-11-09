import copy
import logging
from typing import Dict, Optional, Callable

from ray.rllib.utils.from_config import from_config
from ray.util import log_once

from fllib.clients.callbacks import ClientCallback
from fllib.types import NotProvided, PartialAlgorithmConfigDict
from fllib.types import TYPE_CHECKING

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class ClientConfig:
    def __init__(self, class_specifier) -> None:
        super().__init__()

        self.uid = None
        self.cls = class_specifier

        # self.lr = 0.1
        self.num_batch_per_round = 1

        # `self.dataset()`
        self.dataset_identifier = "mnist"
        self.train_batch_size = 32
        self.eva_batch_size = 64

        # `self.model()`
        self.model_identifier = "mlp"
        self.model_config = None
        self.loss_identifier = "cross_entropy"
        self.loss_kargs = {}

        # `self.training()`
        self.opt_identifier = "sgd"
        self.opt_kargs = {"lr": 0.1}
        self.lr = 0.1
        self.momentum = 0.0
        self.lr_scheduler = None
        self.lr_scheduler_config = None

        self.is_trainable = True
        self.is_testable = True

        # `self.callbacks()`
        self.callbacks_config = ClientCallback

    def training(
        self,
        num_batch_per_round: int = NotProvided,
        lr: float = NotProvided,
        momentum: float = NotProvided,
    ) -> "ClientConfig":
        """Set the number of batches per round.

        Args:
            num_batch_per_round (int): Number of batches per round.

        Returns:
            ClientConfig: This updated ClientConfig object.
        """
        if num_batch_per_round is not NotProvided:
            self.num_batch_per_round = num_batch_per_round
        if lr is not NotProvided:
            self.lr = lr
        if momentum is not NotProvided:
            self.momentum = momentum
        return self

    def trainable(self, trainable: bool = True) -> "ClientConfig":
        self.is_trainable = trainable
        return self

    def testable(self, testable: bool = True) -> "ClientConfig":
        self.is_testable = testable
        return self

    def client_id(self, uid: str) -> "ClientConfig":
        """Set the client id.

        Args:
            uid (str): Client id.

        Returns:
            ClientConfig: This updated ClientConfig object.
        """
        self.uid = uid
        return self

    def callbacks(self, callbacks_config) -> "ClientConfig":
        """Sets the callbacks configuration.

        Args:
            callbacks_config: Callbacks class, whose methods will be run during
                various phases of training and environment sample collection.
                See the `DefaultCallbacks` class and
                `examples/custom_metrics_and_callbacks.py` for more usage information.

        Returns:
            This updated FedavgClientConfig object.
        """
        if callbacks_config is None:
            callbacks_config = ClientCallback
        # Check, whether given `callbacks` is a callable.
        if not callable(callbacks_config):
            raise ValueError(
                "`config.callbacks_config` must be a callable method that "
                "returns a subclass of DefaultCallbacks, got "
                f"{callbacks_config}!"
            )
        self.callbacks_config = callbacks_config

        return self

    def dataset(
        self,
        dataset_identifier: str = NotProvided,
        *,
        train_batch_size: int = NotProvided,
        eva_batch_size: int = NotProvided,
    ) -> "ClientConfig":
        if dataset_identifier is not NotProvided:
            self.dataset_identifier = dataset_identifier
        if train_batch_size is not NotProvided:
            self.train_batch_size = train_batch_size
        if eva_batch_size is not NotProvided:
            self.eva_batch_size = eva_batch_size
        return self

    def model(
        self,
        model_identifier: Optional[str] = NotProvided,
        *,
        model_config: Optional[Dict] = NotProvided,
        loss: Optional[str] = NotProvided,
    ) -> "ClientConfig":
        if model_identifier is not NotProvided:
            self.model_identifier = model_identifier
        if model_config is not NotProvided:
            self.model_config = model_config
        if loss is not NotProvided:
            self.loss = loss
        return self

    def optimizer(
        self,
        opt_identifier: Optional[str] = NotProvided,
        *,
        opt_kargs: Optional[Dict] = NotProvided,
        lr_scheduler: Optional[str] = NotProvided,
        lr_scheduler_config: Optional[Dict] = NotProvided,
    ) -> "ClientConfig":
        if opt_identifier is not NotProvided:
            self.opt_identifier = opt_identifier
        if opt_kargs is not NotProvided:
            self.opt_kargs = opt_kargs
        if lr_scheduler is not NotProvided:
            self.lr_scheduler = lr_scheduler
        if lr_scheduler_config is not NotProvided:
            self.lr_scheduler_config = lr_scheduler_config
        return self

    def build(self, use_copy: bool = True) -> Callable:
        """Build the client.

        Args:
            use_copy (bool, optional): Defaults to True.

        Returns:
            Callable: the instantiated client.
        """
        client_cls = self.cls
        # if isinstance(client_cls, str):
        #     raise ValueError("string `client_cls` is not supported in FLlib yet")
        if client_cls is None:
            raise ValueError("client_cls is not set")

        return from_config(
            client_cls,
            config=None,
            **{"client_config": self if not use_copy else copy.deepcopy(self)},
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
    ) -> "ClientConfig":
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

            if key in ["num_batch_per_round", "lr", "momentum"]:
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
