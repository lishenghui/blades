import random

import torch

from fedlib.constants import CLIENT_UPDATE, CLIENT_ID
from fedlib.trainers import Trainer as Algorithm
from fedlib.trainers import Trainer
from fedlib.aggregators import Signguard

from .adversary import Adversary


class ALIEAdversary(Adversary):
    def on_trainer_init(self, trainer: Trainer):
        super().on_trainer_init(trainer)
        
        self.num_clients = trainer.config.num_clients
        num_byzantine = len(self.clients)

        s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
        cdf_value = (self.num_clients - num_byzantine - s) / (
            self.num_clients - num_byzantine
        )
        dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.z_max = dist.icdf(cdf_value)

        self.negative_indices = None

    def on_local_round_end(self, trainer: Algorithm):
        benign_updates = self.get_benign_updates(trainer)
        mean = benign_updates.mean(dim=0)
        std = benign_updates.std(dim=0)

        # For SignGuard, we need to negate some of the elements
        if isinstance(trainer.server.aggregator, Signguard):
            if self.negative_indices is None:
                num_elements = len(std)
                num_negate = num_elements // 2
                self.negative_indices = random.sample(range(num_negate), num_negate)

            std[self.negative_indices] *= -1

        update = mean + std * self.z_max
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update
