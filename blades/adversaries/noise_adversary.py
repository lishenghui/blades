from typing import Optional

import torch

from fedlib.trainers import Trainer
from fedlib.constants import CLIENT_UPDATE, TRAIN_LOSS, CLIENT_ID
from .adversary import Adversary


class NoiseAdversary(Adversary):
    def __init__(
        self,
        mean: Optional[float] = 0.1,
        std: Optional[float] = 0.1,
    ):
        super().__init__()

        self._noise_mean = mean
        self._noise_std = std

    def on_local_round_end(self, trainer: Trainer):
        benign_updates = self.get_benign_updates(trainer)
        mean = benign_updates.mean(dim=0)

        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                device = mean.device
                noise = torch.normal(
                    self._noise_mean, self._noise_std, size=mean.shape
                ).to(device)
                result[CLIENT_UPDATE] = noise
                result[TRAIN_LOSS] = -1
