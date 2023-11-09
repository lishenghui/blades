import random
from typing import Dict

import torch

from fllib.aggregators import Signguard
from fllib.algorithms import Algorithm
from fllib.constants import CLIENT_UPDATE
from .adversary import Adversary


class ALIEAdversary(Adversary):
    def __init__(self, clients, global_config: Dict = None):
        super().__init__(clients, global_config)

        self.num_clients = global_config.num_clients
        num_byzantine = len(clients)

        s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
        cdf_value = (self.num_clients - num_byzantine - s) / (
            self.num_clients - num_byzantine
        )
        dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.z_max = dist.icdf(cdf_value)

        self.negative_indices = None

    def on_local_round_end(self, algorithm: Algorithm):
        benign_updates = self.get_benign_updates(algorithm)
        mean = benign_updates.mean(dim=0)
        std = benign_updates.std(dim=0)

        # For SignGuard, we need to negate some of the elements
        if isinstance(algorithm.server.aggregator, Signguard):
            if self.negative_indices is None:
                num_elements = len(std)
                num_negate = num_elements // 2
                self.negative_indices = random.sample(range(num_negate), num_negate)

            std[self.negative_indices] *= -1

        update = mean + std * self.z_max
        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result["id"])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update
