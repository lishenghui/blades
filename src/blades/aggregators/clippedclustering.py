from typing import List, Union

import numpy as np
import torch
from numpy import inf
from sklearn.cluster import AgglomerativeClustering, KMeans

from blades.clients.client import BladesClient
from blades.utils import torch_utils
from .mean import Mean, _BaseAggregator
from .median import Median


class Clippedclustering(_BaseAggregator):
    r"""A robust aggregator from paper `An Experimental Study of Byzantine-
    Robust Aggregation Schemes in Federated L earning.

    <https://doi.org/10.36227/techrxiv.19560325.v1>`_. It separates the client
    population into two groups based on the cosine similarities.

    Args:
        tau (float): threshold of clipping norm.
                    If it is not \given, updates are clipped according the median of
                    historical norm.

    __
    """

    def __init__(
        self, agg="mean", signguard=False, max_tau=1e5, linkage="average"
    ) -> None:
        super(Clippedclustering, self).__init__()

        assert linkage in ["average", "single"]
        self.tau = max_tau
        self.signguard = signguard
        self.linkage = linkage
        self.l2norm_his = []
        if agg == "mean":
            self.agg = Mean()
        elif agg == "median":
            self.agg = Median()
        else:
            raise NotImplementedError(f"{agg} is not supported yet.")

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
        l2norms = [torch.norm(update).item() for update in updates]
        self.l2norm_his.extend(l2norms)
        threshold = np.median(self.l2norm_his)
        threshold = min(threshold, self.tau)

        # print(threshold, l2norms)
        for idx, l2 in enumerate(l2norms):
            if l2 > threshold:
                updates[idx] = torch_utils.clip_tensor_norm_(updates[idx], threshold)

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
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage=self.linkage, n_clusters=2
        )
        clustering.fit(dis_max)

        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        S1_idxs = list(
            [idx for idx, label in enumerate(clustering.labels_) if label == flag]
        )
        selected_idxs = S1_idxs

        if self.signguard:
            features = []
            num_para = len(updates[0])
            for update in updates:
                feature0 = (update > 0).sum().item() / num_para
                feature1 = (update < 0).sum().item() / num_para
                feature2 = (update == 0).sum().item() / num_para

                features.append([feature0, feature1, feature2])

            kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

            flag = 1 if np.sum(kmeans.labels_) > num // 2 else 0
            S2_idxs = list(
                [idx for idx, label in enumerate(kmeans.labels_) if label == flag]
            )

            selected_idxs = list(set(S1_idxs) & set(S2_idxs))

        benign_updates = []
        for idx in selected_idxs:
            benign_updates.append(updates[idx])

        values = self.agg(torch.vstack(benign_updates))
        return values
