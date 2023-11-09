from typing import Dict

import numpy as np
import torch
from numpy import inf
from sklearn.cluster import AgglomerativeClustering

from fllib.aggregators import Clippedclustering
from fllib.algorithms import Algorithm
from fllib.constants import CLIENT_UPDATE
from .adversary import Adversary


class AttackclippedclusteringAdversary(Adversary):
    def __init__(self, clients, global_config: Dict = None, linkage="single"):
        super().__init__(clients, global_config)
        self.linkage = linkage
        self.agg = Clippedclustering()

    def on_local_round_end(self, algorithm: Algorithm):
        self._chain_attack(algorithm)

    def _chain_attack(self, algorithm):
        benign_updates = self.get_benign_updates(algorithm)
        benign_mean = benign_updates.mean(dim=0)

        num = len(benign_updates)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - torch.nn.functional.cosine_similarity(
                    benign_updates[i, :], benign_updates[j, :], dim=0
                )
                dis_max[j, i] = dis_max[i, j]
        dis_max[dis_max == -inf] = 0
        dis_max[dis_max == inf] = 2
        dis_max[np.isnan(dis_max)] = 2
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="single", n_clusters=2
        )
        clustering.fit(dis_max)

        dis_cross = inf
        for idx_i, _ in enumerate(clustering.labels_):
            for idx_j, _ in enumerate(clustering.labels_):
                if idx_j == idx_i:
                    continue
                dis = 1 - torch.nn.functional.cosine_similarity(
                    benign_updates[idx_i, :], benign_updates[idx_j, :], dim=0
                )
                dis_cross = min(dis_cross, dis)
        theta_cross = torch.acos(1 - dis_cross) - 0.1

        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0

        larger_group = torch.vstack(
            list(
                model / torch.norm(model)
                for model, label in zip(benign_updates, clustering.labels_)
                if label == flag
            )
        )
        dis2mean = torch.tensor(
            [
                1
                - torch.nn.functional.cosine_similarity(
                    benign_update, benign_mean, dim=0
                )
                for benign_update in larger_group
            ]
        )
        idx_max_dis = torch.argmax(dis2mean)
        theta = torch.acos(1 - dis2mean[idx_max_dis])
        mal_update = benign_updates[idx_max_dis] / torch.norm(
            benign_updates[idx_max_dis]
        )
        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result["id"])
            if client.is_malicious:
                if theta + theta_cross >= np.pi:
                    mal_update = -10 * benign_updates.mean(dim=0)
                else:
                    a = torch.cos(theta + theta_cross - 1e-4) - torch.sin(
                        theta + theta_cross - 1e-4
                    ) / torch.tan(theta)
                    b = torch.cos(theta_cross - 1e-4) + torch.sin(
                        theta_cross - 1e-4
                    ) / torch.tan(theta)
                    mal_update = 10 * (
                        a
                        * benign_updates.mean(dim=0)
                        / torch.norm(benign_updates.mean(dim=0))
                        + b * mal_update
                    )
                    theta = theta + theta_cross - 1e-4
                result[CLIENT_UPDATE] = mal_update
