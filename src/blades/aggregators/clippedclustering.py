'''
    S. Li, C. H. E. Ngai and T. Voigt,“An Experimental Study of Byzantine-Robust Aggregation Schemes in Federated Learning”.
    TechRxiv,11-Apr-2022,doi:10.36227/techrxiv.19560325.v1.
'''
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
    def __init__(self) -> None:
        self.l2norm_his = []
    
    def __call__(self, inputs):
        l2norms = [torch.norm(update).item() for update in inputs]
        self.l2norm_his.extend(l2norms)
        threshold = np.median(self.l2norm_his)
        # threshold = min(threshold, 5.0)
        
        for idx, l2 in enumerate(l2norms):
            if l2 > threshold:
                inputs[idx] = torch_utils.clip_tensor_norm_(inputs[idx], threshold)
        
        stacked_models = torch.vstack(inputs)
        np_models = stacked_models.cpu().detach().numpy()
        num = len(inputs)
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
        values = torch.vstack(list(model for model, label in zip(inputs, clustering.labels_) if label == flag)).mean(
            dim=0)
        return values
