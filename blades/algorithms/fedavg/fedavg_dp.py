import math

from ray.rllib.utils.annotations import override

from blades.clients.dp_client import DPCliengConfig
from fllib.algorithms import AlgorithmConfig
from .fedavg import FedavgConfig, Fedavg


class FedavgDPConfig(FedavgConfig):
    def __init__(self, algo_class=None):
        """Initializes a FedavgConfig instance."""
        super().__init__(algo_class=algo_class or FedavgDP)

        self.dp_privacy_delta = 1e-6
        self.dp_privacy_epsilon = 1.0
        self.dp_clip_threshold = 1.0

        self.dp_privacy_sensitivity = 0

    def get_client_config(self) -> DPCliengConfig:
        noise_factor = (
            self.dp_privacy_sensitivity
            * math.sqrt(2 * math.log(1.25 / self.dp_privacy_delta))
            / self.dp_privacy_epsilon
        )
        config = (
            DPCliengConfig()
            .training(
                num_batch_per_round=self.num_batch_per_round,
                # lr=self.local_lr,
                clip_threshold=self.dp_clip_threshold,
                noise_factor=noise_factor,
            )
            .update_from_dict(self.client_config)
        )
        return config

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()
        self.dp_privacy_sensitivity = (
            2 * self.dp_clip_threshold / self.dataset_config["train_bs"]
        )


class FedavgDP(Fedavg):
    def __init__(self, config=None, logger_creator=None, **kwargs):
        super().__init__(config, logger_creator, **kwargs)

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return FedavgDPConfig()
