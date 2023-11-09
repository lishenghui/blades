from typing import Dict, Optional

import torch

from fllib.algorithms import Algorithm
from fllib.constants import CLIENT_UPDATE
from .adversary import Adversary


class NoiseAdversary(Adversary):
    def __init__(
        self,
        clients,
        global_config: Dict = None,
        mean: Optional[float] = 0.1,
        std: Optional[float] = 0.1,
    ):
        super().__init__(clients, global_config)

        self._noise_mean = mean
        self._noise_std = std

    def on_local_round_end(self, algorithm: Algorithm):
        benign_updates = self.get_benign_updates(algorithm)
        mean = benign_updates.mean(dim=0)
        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result["id"])
            if client.is_malicious:
                device = mean.device
                noise = torch.normal(
                    self._noise_mean, self._noise_std, size=mean.shape
                ).to(device)
                result[CLIENT_UPDATE] = noise
