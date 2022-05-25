import numpy as np
import torch
from numpy import inf
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering


class Clustering():
    def __init__(self, options) -> None:
        pass
    
    def __call__(self, inputs):
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
        # with open('../notebooks/updates_fedsgd_ipm.npy', 'wb') as f:
        #     np.save(f, dis_max)
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=2)
        clustering.fit(dis_max)
        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        values = torch.vstack(list(model for model, label in zip(inputs, clustering.labels_) if label == flag)).mean(
            dim=0)
        return values
