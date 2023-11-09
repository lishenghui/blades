from typing import Dict

import torch

from fllib.algorithms import Algorithm
from fllib.constants import CLIENT_UPDATE
from .adversary import Adversary


class AdaptiveAdversary(Adversary):
    def __init__(self, clients, global_config: Dict = None):
        super().__init__(clients, global_config)

    def on_local_round_end(self, algorithm: Algorithm):
        updates = self._attack_median_and_trimmedmean(algorithm)
        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result["id"])
            if client.is_malicious:
                result[CLIENT_UPDATE] = updates

        return updates

    def _attack_median_and_trimmedmean(self, algorithm: Algorithm):
        benign_updates = self.get_benign_updates(algorithm)
        device = benign_updates.device
        mean_grads = benign_updates.mean(dim=0)
        deviation = torch.sign(mean_grads).to(device)
        max_vec, _ = benign_updates.max(dim=0)
        min_vec, _ = benign_updates.min(dim=0)
        b = 2

        neg_pos_mask = torch.logical_and(deviation == -1, max_vec > 0)
        neg_neg_mask = torch.logical_and(deviation == -1, max_vec < 0)
        pos_pos_mask = torch.logical_and(deviation == 1, min_vec > 0)
        pos_neg_mask = torch.logical_and(deviation == 1, min_vec < 0)
        zero_mask = deviation == 0

        # Compute the result for different conditions using tensor operations
        rand_neg_pos = torch.rand(neg_pos_mask.sum(), device=device)
        rand_neg_max = torch.rand(neg_neg_mask.sum(), device=device)
        rand_pos_min = torch.rand(pos_pos_mask.sum(), device=device)
        rand_pos_neg = torch.rand(pos_neg_mask.sum(), device=device)

        neg_pos_max = (
            rand_neg_pos * ((b - 1) * max_vec[neg_pos_mask]) + max_vec[neg_pos_mask]
        )
        neg_neg_max = (
            rand_neg_max * ((1 / b - 1) * max_vec[neg_neg_mask]) + max_vec[neg_neg_mask]
        )
        pos_pos_min = (
            rand_pos_min * ((1 - 1 / b) * min_vec[pos_pos_mask])
            + min_vec[pos_pos_mask] / b
        )
        pos_neg_min = (
            rand_pos_neg * ((1 - b) * min_vec[pos_neg_mask]) + min_vec[pos_neg_mask] * b
        )
        result_zero = mean_grads[zero_mask].repeat(1)

        # Combine the results
        result = torch.zeros_like(mean_grads)
        result[neg_pos_mask] = neg_pos_max
        result[neg_neg_mask] = neg_neg_max
        result[pos_pos_mask] = pos_pos_min
        result[pos_neg_mask] = pos_neg_min
        result[zero_mask] = result_zero

        return result
