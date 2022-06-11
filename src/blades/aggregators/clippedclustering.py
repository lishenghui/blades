import os
import sys

import numpy as np
import torch
from numpy import inf
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import torch_utils


class Clippedclustering():
    r"""
         A robust aggregator from paper `"An Experimental Study of Byzantine-Robust sAggregation Schemes in Federated Learning" <https://www.techrxiv.org/articles/preprint/An_Experimental_Study_of_Byzantine-Robust_Aggregation_Schemes_in_Federated_Learning/19560325>`_

         it separates the client population into two groups based on the cosine similarities
    """
    
    def __init__(self) -> None:
        self.l2norm_his = []
    
    def __call__(self, clients):
        updates = list(map(lambda w: w.get_update(), clients))
        l2norms = [torch.norm(update).item() for update in updates]
        self.l2norm_his.extend(l2norms)
        threshold = np.median(self.l2norm_his)
        # threshold = min(threshold, 5.0)
        
        for idx, l2 in enumerate(l2norms):
            if l2 > threshold:
                updates[idx] = torch_utils.clip_tensor_norm_(updates[idx], threshold)
        
        stacked_models = torch.vstack(updates)
        np_models = stacked_models.cpu().detach().numpy()
        num = len(updates)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                if i == j:
                    dis_max[i, j] = 1
                else:
                    dis_max[i, j] = 1 - spatial.distance.cosine(np_models[i, :], np_models[j, :])
        dis_max[dis_max == -inf] = -1
        dis_max[dis_max == inf] = 1
        dis_max[np.isnan(dis_max)] = -1
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=2)
        clustering.fit(dis_max)
        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        values = torch.vstack(list(model for model, label in zip(updates, clustering.labels_) if label == flag)).mean(
            dim=0)
        return values
