from typing import List

import numpy as np
import torch
from numpy import inf
from sklearn.cluster import AgglomerativeClustering

from fllib.utils import torch_utils
from .aggregators import Mean, Median


class Clippedclustering(object):
    def __init__(self, agg="mean", max_tau=1e5, linkage="average") -> None:
        super(Clippedclustering, self).__init__()

        assert linkage in ["average", "single"]
        self.tau = max_tau
        self.linkage = linkage
        self.l2norm_his = []
        if agg == "mean":
            self.agg = Mean()
        elif agg == "median":
            self.agg = Median()
        else:
            raise NotImplementedError(f"{agg} is not supported yet.")

    def __call__(self, inputs: List[torch.Tensor]):
        # Clip updates based on L2 norm
        updates = self._clip_updates(inputs)

        # Compute pairwise cosine similarity
        dis_max = self._compute_cosine_similarity(updates)

        # Cluster updates using AgglomerativeClustering
        selected_idxs = self._cluster_updates(dis_max)
        # Compute final values using selected updates
        values = self._compute_values(selected_idxs, updates)

        return values

    def _clip_updates(self, inputs: List[torch.Tensor]):
        l2norms = [torch.norm(update).item() for update in inputs]
        self.l2norm_his.extend(l2norms)
        threshold = np.median(self.l2norm_his)
        threshold = min(threshold, self.tau)

        clipped_updates = []
        for idx, l2 in enumerate(l2norms):
            if l2 > threshold:
                clipped_updates.append(
                    torch_utils.clip_tensor_norm_(inputs[idx], threshold)
                )
            else:
                clipped_updates.append(inputs[idx])

        return torch.stack(clipped_updates, dim=0)

    def _compute_cosine_similarity(self, updates):
        num = len(updates)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - torch.nn.functional.cosine_similarity(
                    updates[i, :], updates[j, :], dim=0
                )
                dis_max[j, i] = dis_max[i, j]
        dis_max[dis_max == -inf] = 0
        dis_max[dis_max == inf] = 2
        dis_max[np.isnan(dis_max)] = 2
        return dis_max

    def _cluster_updates(self, dis_max):
        clustering = AgglomerativeClustering(
            metric="precomputed", linkage=self.linkage, n_clusters=2
        )
        clustering.fit(dis_max)

        flag = 1 if np.sum(clustering.labels_) > len(dis_max) // 2 else 0
        selected_idxs = [
            idx for idx, label in enumerate(clustering.labels_) if label == flag
        ]

        return selected_idxs

    def _compute_values(self, selected_idxs, updates):
        benign_updates = []
        for idx in selected_idxs:
            benign_updates.append(updates[idx])

        values = self.agg(benign_updates)
        return values
