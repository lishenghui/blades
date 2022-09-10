import os
import sys
from typing import Union, List

import numpy as np
import torch
from sklearn.cluster import KMeans

from blades.client import BladesClient
from .mean import _BaseAggregator, Mean
from .median import Median

from blades.utils import torch_utils


class Dnc(_BaseAggregator):
    r"""
         A robust aggregator from paper `"An Experimental Study of Byzantine-Robust sAggregation Schemes in Federated Learning" <https://www.techrxiv.org/articles/preprint/An_Experimental_Study_of_Byzantine-Robust_Aggregation_Schemes_in_Federated_Learning/19560325>`_

         it separates the client population into two groups based on the cosine similarities

    Args:
        tau (float): threshold of clipping norm. If it is not given, updates are clipped according the median of historical norm.
    """
    
    def __init__(self, num_byzantine, *, sub_dim=10000, num_iters=1, filter_frac=1.0) -> None:
        super(Dnc, self).__init__()
        
        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.fliter_frac = filter_frac
    
    def __call__(self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]):
        updates = self._get_updates(inputs)
        d = len(updates[0])
        l2norms = [torch.norm(update).item() for update in updates]
        threshold = np.median(l2norms)

        # for idx, l2 in enumerate(l2norms):
        #     if l2 > threshold:
        #         updates[idx] = torch_utils.clip_tensor_norm_(updates[idx], threshold)
                
        benign_ids = []
        for i in range(self.num_iters):
            indices = torch.randperm(d)[:self.sub_dim]
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
            s = np.array([(torch.dot(update - mu, v) ** 2).item() for update in sub_updates])

            good = s.argsort()[:len(updates) - int(self.fliter_frac * self.num_byzantine)]
            benign_ids.extend(good)
            
            # full_cov = sub_updates.cpu().numpy()
            # full_mean = mu.cpu().numpy()
            # centered_cov = full_cov - full_mean
            # u, s, vh = np.linalg.svd(centered_cov, full_matrices=False)
            # # print('Top 7 Singular Values: ', s[0:7])
            # eigs = vh[0:1]
            # corrs = np.matmul(eigs, np.transpose(full_cov))  # shape num_top, num_active_indices
            # scores = np.linalg.norm(corrs, axis=0)  # shape num_active_indices
        benign_ids = list(set(benign_ids))
        benign_updates = updates[benign_ids, :].mean(dim=0)
        return benign_updates
