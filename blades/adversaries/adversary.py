import copy
from typing import Dict, Type, List

import torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.from_config import from_config

from fedlib.constants import CLIENT_UPDATE, CLIENT_ID
from fedlib.trainers import FedavgTrainer, Trainer, TrainerCallback


class Adversary(TrainerCallback):
    """Base class for all adversaries."""

    def get_benign_updates(self, algorithm) -> torch.Tensor:
        """Returns a tensor of benign updates from the local results."""
        updates = []
        for result in algorithm.local_results:
            psudo_grad = result.get(CLIENT_UPDATE, None)
            client = algorithm.client_manager.get_client_by_id(result[CLIENT_ID])
            if psudo_grad is not None and not client.is_malicious:
                updates.append(psudo_grad)

        if not updates:
            raise ValueError(
                "No benign updates found. To use this adversary,"
                "you must have at least one benign client."
            )
        return torch.vstack(updates)

    @override(TrainerCallback)
    def setup(self, trainer: Trainer, **info):
        super().setup(trainer, **info)

        num_malicious = trainer.config.num_malicious_clients
        self.clients = trainer.client_manager.clients[:num_malicious]

    @override(TrainerCallback)
    def on_trainer_init(self, trainer: FedavgTrainer):
        """Called when the algorithm starts."""
        for client in self.clients:
            client.to_malicious(local_training=False)
        _ = trainer  # Mute "unused argument" warning

    def on_local_round_end(self, trainer: FedavgTrainer):
        """Called when the local round ends."""


class AdversaryConfig:
    """Base class for all adversary configs."""

    def __init__(self, adversary_cls=Adversary) -> None:
        self.adversary_class: Type["Adversary"] = adversary_cls

    def to_dict(self) -> dict:
        """Converts all settings into a legacy config dict for backward
        compatibility.

        Returns:
            A complete AlgorithmConfigDict, usable in backward-compatible Tune/RLlib
            use cases, e.g. w/ `tune.Tuner().fit()`.
        """
        config = copy.deepcopy(vars(self))

        return config

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
        return getattr(self, item)

    def get(self, key, default=None):
        """Shim method to help pretend we are a dict."""
        # prop = self._translate_special_keys(key)
        return getattr(self, key, default)

    def build(self, clients: List) -> "Adversary":
        """Builds the Adversary instance."""
        config_dict = self.to_dict()
        cls = config_dict.pop("adversary_class", Adversary)
        return from_config(cls, config_dict, clients=clients)

    def pre_build(self) -> "Adversary":
        """Builds the Adversary instance."""

        config_dict = self.to_dict()
        config_dict["type"] = config_dict.pop("adversary_class", Adversary)
        return config_dict
        # cls = config_dict.pop("adversary_class", Adversary)
        # partial_cls = partial(cls, **config_dict)
        # partial_cls = from_config(cls, **config_dict)
        # breakpoint()

        # return partial_cls

    def update_from_dict(
        self,
        config_dict: Dict,
    ) -> "AdversaryConfig":
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
            if key == "type":
                key = "adversary_class"
            setattr(self, key, value)
        return self
