import random
from typing import Dict

import torch

from fllib.aggregators import Signguard
from fllib.algorithms import Algorithm
from fllib.constants import CLIENT_UPDATE
from .adversary import Adversary


class MinMaxAdversary(Adversary):
    def __init__(self, clients, global_config: Dict = None):
        super().__init__(clients, global_config)

        self.threshold = 3.0
        self.threshold_diff = 1e-4
        self.num_byzantine = None
        self.negative_indices = None

    def on_local_round_end(self, algorithm: Algorithm):
        if self.num_byzantine is None:
            self.num_byzantine = 0
            for result in algorithm.local_results:
                client = algorithm.client_manager.get_client_by_id(result["id"])
                if client.is_malicious:
                    self.num_byzantine += 1

        updates = self._attack_by_binary_search(algorithm)
        # updates = self._attack_median_and_trimmedmean(algorithm)
        self.num_byzantine = 0
        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result["id"])
            if client.is_malicious:
                result[CLIENT_UPDATE] = updates
                self.num_byzantine += 1
        return updates

    def _attack_by_binary_search(self, algorithm: Algorithm):
        benign_updates = self.get_benign_updates(algorithm)
        mean_grads = benign_updates.mean(dim=0)
        deviation = benign_updates.std(dim=0)
        threshold = torch.cdist(benign_updates, benign_updates, p=2).max()

        # For SignGuard, we need to negate some of the elements
        if isinstance(algorithm.server.aggregator, Signguard):
            if self.negative_indices is None:
                num_elements = len(deviation)
                num_negate = num_elements // 2
                self.negative_indices = random.sample(range(num_negate), num_negate)
                self.negative_indices = random.sample(range(num_negate), num_negate)

            deviation[self.negative_indices] *= -1

        low = 0
        high = 5
        while abs(high - low) > 0.01:
            mid = (low + high) / 2
            mal_update = torch.stack([mean_grads - mid * deviation])
            loss = torch.cdist(mal_update, benign_updates, p=2).max()
            if loss < threshold:
                low = mid
            else:
                high = mid
        return mean_grads - mid * deviation
