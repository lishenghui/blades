import os
import sys
from typing import Union, List

import numpy as np
import torch
from numpy import inf
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering

from blades.client import BladesClient
from .mean import _BaseAggregator

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import torch_utils


class Clippedclustering(_BaseAggregator):
    r"""
         A robust aggregator from paper `"An Experimental Study of Byzantine-Robust sAggregation Schemes in Federated Learning" <https://www.techrxiv.org/articles/preprint/An_Experimental_Study_of_Byzantine-Robust_Aggregation_Schemes_in_Federated_Learning/19560325>`_

         it separates the client population into two groups based on the cosine similarities

    Args:
        tau (float): threshold of clipping norm. If it is not given, updates are clipped according the median of historical norm.
    """
    
    def __init__(self, tau=None) -> None:
        super(Clippedclustering, self).__init__()
        self.tau = tau
        self.l2norm_his = []
    
    def __call__(self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]):
        updates = self._get_updates(inputs)
        l2norms = [torch.norm(update).item() for update in updates]
        self.l2norm_his.extend(l2norms)
        if self.tau:
            threshold = self.tau
        else:
            threshold = np.median(self.l2norm_his)
        
        for idx, l2 in enumerate(l2norms):
            if l2 > threshold:
                updates[idx] = torch_utils.clip_tensor_norm_(updates[idx], threshold)
        
        # stacked_models = torch.vstack(updates)
        np_models = updates.cpu().detach().numpy()
        num = len(updates)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                if i == j:
                    dis_max[i, j] = 0
                else:
                    dis_max[i, j] = spatial.distance.cosine(np_models[i, :], np_models[j, :])
        dis_max[dis_max == -inf] = 0
        dis_max[dis_max == inf] = 2
        dis_max[np.isnan(dis_max)] = 2
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='complete', n_clusters=2)
        clustering.fit(dis_max)
        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        values = torch.vstack(list(model for model, label in zip(updates, clustering.labels_) if label == flag)).mean( dim=0)
        return values
