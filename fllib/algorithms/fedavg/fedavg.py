from typing import Optional

from fllib.algorithms.algorithm import Algorithm
from fllib.algorithms.algorithm_config import AlgorithmConfig
from fllib.types import NotProvided


class FedavgConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        """Initializes a FedavgConfig instance."""
        super().__init__(algo_class=algo_class or Fedavg)

        # self.adversarial()`
        self.num_malicious_clients = 0
        self.attack_type = None

    def adversary(
        self,
        *,
        num_malicious_clients: Optional[int] = NotProvided,
        attack_type=NotProvided
    ):
        if num_malicious_clients is not NotProvided:
            self.num_malicious_clients = num_malicious_clients
        if attack_type is not NotProvided:
            self.attack_type = attack_type
        return self


class Fedavg(Algorithm):
    def __init__(self, config=None, logger_creator=None, **kwargs):
        super().__init__(config, logger_creator, **kwargs)

    def setup(self, config: AlgorithmConfig):
        super().setup(config)

    def training_step(self):
        pass

    @classmethod
    def get_default_config(cls):
        return AlgorithmConfig()

    def evaluate(self):
        pass
