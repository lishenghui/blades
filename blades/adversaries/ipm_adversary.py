from typing import Dict

from fllib.algorithms import Algorithm
from fllib.constants import CLIENT_UPDATE
from .adversary import Adversary


class IPMAdversary(Adversary):
    def __init__(self, clients, global_config: Dict = None, scale: float = 1.0):
        super().__init__(clients, global_config)

        self._scale = scale

    def on_local_round_end(self, algorithm: Algorithm):
        benign_updates = self.get_benign_updates(algorithm)
        mean = benign_updates.mean(dim=0)

        update = -self._scale * mean
        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result["id"])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update
