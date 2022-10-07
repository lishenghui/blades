from typing import List, Union

import numpy as np
import torch
from numpy import inf
from sklearn.cluster import AgglomerativeClustering

from blades.clients.client import BladesClient
from .mean import _BaseAggregator


class Clustering(_BaseAggregator):
    r"""A robust aggregator from paper `On the byzantine robustness of clustered
    federated learning.

    <https://ieeexplore.ieee.org/abstract/document/9054676>`_.

    It separates the client population into two groups based on the cosine
    similarities.
    """

    def __init__(self):
        super(Clustering, self).__init__()

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
        num = len(updates)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - torch.nn.functional.cosine_similarity(
                    updates[i, :], updates[j, :], dim=0
                )
                dis_max[j, i] = dis_max[i, j]
        dis_max[dis_max == -inf] = -1
        dis_max[dis_max == inf] = 1
        dis_max[np.isnan(dis_max)] = -1
        # with open('../notebooks/updates_fedsgd_ipm.npy', 'wb') as f:
        #     np.save(f, dis_max)
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="complete", n_clusters=2
        )
        clustering.fit(dis_max)
        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        values = torch.vstack(
            list(
                model
                for model, label in zip(updates, clustering.labels_)
                if label == flag
            )
        ).mean(dim=0)
        return values
