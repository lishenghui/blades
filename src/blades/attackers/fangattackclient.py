from typing import Generator

import numpy as np
import torch

from blades.aggregators.multikrum import Multikrum
from blades.clients.client import ByzantineClient


class FangattackClient(ByzantineClient):
    def omniscient_callback(self, simulator):
        pass

    def train_global_model(self, train_set: Generator, num_batches: int, opt) -> None:
        pass
        pass


class FangattackAdversary:
    r""""""

    def __init__(
        self,
        num_byzantine: int,
        agg: str,
        dev_type="std",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dev_type = dev_type
        self.agg = agg
        self.num_byzantine = num_byzantine

    def attack_median_and_trimmedmean(self, simulator):
        benign_update = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                benign_update.append(w.get_update())
        benign_update = torch.stack(benign_update, 0)
        agg_grads = torch.mean(benign_update, 0)
        deviation = torch.sign(agg_grads)
        device = benign_update.device
        b = 2
        max_vector = torch.max(benign_update, 0)[0]
        min_vector = torch.min(benign_update, 0)[0]

        max_ = (max_vector > 0).type(torch.FloatTensor).to(device)
        min_ = (min_vector < 0).type(torch.FloatTensor).to(device)

        max_[max_ == 1] = b
        max_[max_ == 0] = 1 / b
        min_[min_ == 1] = b
        min_[min_ == 0] = 1 / b

        max_range = torch.cat(
            (max_vector[:, None], (max_vector * max_)[:, None]), dim=1
        )
        min_range = torch.cat(
            ((min_vector * min_)[:, None], min_vector[:, None]), dim=1
        )

        rand = (
            torch.from_numpy(
                np.random.uniform(0, 1, [len(deviation), self.num_byzantine])
            )
            .type(torch.FloatTensor)
            .to(benign_update.device)
        )

        max_rand = (
            torch.stack([max_range[:, 0]] * rand.shape[1]).T
            + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        )
        min_rand = (
            torch.stack([min_range[:, 0]] * rand.shape[1]).T
            + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
        )

        mal_vec = (
            torch.stack(
                [(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]
            ).T.to(device)
            * max_rand
            + torch.stack(
                [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]
            ).T.to(device)
            * min_rand
        ).T

        for i, client in enumerate(simulator._clients.values()):
            if client.is_byzantine():
                client.save_update(mal_vec[i])

    def attack_multikrum(self, simulator):
        multi_krum = Multikrum(num_excluded=self.num_byzantine, k=1)
        benign_update = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                benign_update.append(w.get_update())
        benign_update = torch.stack(benign_update, 0)
        agg_updates = torch.mean(benign_update, 0)
        deviation = torch.sign(agg_updates)

        def compute_lambda(all_updates, model_re, n_attackers):

            distances = []
            n_benign, d = all_updates.shape
            for update in all_updates:
                distance = torch.norm((all_updates - update), dim=1)
                distances = (
                    distance[None, :]
                    if not len(distances)
                    else torch.cat((distances, distance[None, :]), 0)
                )

            distances[distances == 0] = 10000
            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(
                distances[:, : n_benign - 2 - n_attackers],
                dim=1,
            )
            min_score = torch.min(scores)
            term_1 = min_score / (
                (n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0]
            )
            max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (
                torch.sqrt(torch.Tensor([d]))[0]
            )

            return term_1 + max_wre_dist

        all_updates = torch.stack(
            list(map(lambda w: w.get_update(), simulator._clients.values()))
        )
        lambda_ = compute_lambda(all_updates, agg_updates, self.num_byzantine)

        threshold = 1e-5
        mal_update = []

        while lambda_ > threshold:
            mal_update = -lambda_ * deviation
            mal_updates = torch.stack([mal_update] * self.num_byzantine)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            # print(mal_updates.shape, n_attackers)
            agg_grads, krum_candidate = multi_krum(mal_updates)
            if krum_candidate < self.num_byzantine:
                return mal_update
            else:
                mal_update = []

            lambda_ *= 0.5

        if not len(mal_update):
            mal_update = agg_updates - lambda_ * deviation

        return mal_update

    def omniscient_callback(self, simulator):
        if self.agg in ["median", "trimmedmean"]:
            self.attack_median_and_trimmedmean(simulator)
        else:
            raise NotImplementedError(
                f"Adaptive attacks to {self.agg} " f"is not supported yet."
            )
