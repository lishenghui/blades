from typing import Union, Optional, List

import numpy as np
import torch

from blades.client import BladesClient
from .mean import _BaseAggregator


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def smoothed_weiszfeld(weights, alphas, z, eps=1e-6, T=5):
    m = len(weights)
    if len(alphas) != m:
        raise ValueError
    
    if eps < 0:
        raise ValueError
    
    for t in range(T):
        betas = []
        for k in range(m):
            distance = _compute_euclidean_distance(z, weights[k])
            betas.append(max(eps, alphas[k] / max(distance, eps)))
        
        z = 0
        for w, beta in zip(weights, betas):
            z += w * beta
        z /= sum(betas)
    return z


class Geomed(_BaseAggregator):
    r"""

    A robust aggregator from paper `"Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent" <https://arxiv.org/abs/1705.05491>`_
    
    ``GeoMed`` aims to find a vector that minimizes the sum of its Euclidean distances to all the update vectors:
    
    .. math::
        GeoMed := \arg\min_{\boldsymbol{z}}  \sum_{k \in [K]} \lVert \boldsymbol{z} -  {\Delta}_i \rVert.
    
    
    There is no closed-form solution to the ``GeoMed`` problem. It is approximately solved using
    Weiszfeld's algorithm in this implementation to.
    
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param eps: Smallest allowed value of denominator, to avoid divide by zero.
    	Equivalently, this is a smoothing parameter. Default 1e-6.
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-10.
    """
    
    def __init__(self, maxiter: Optional[int] = 100, eps: Optional[float] = 1e-6, ftol: Optional[float] = 1e-10):
        self.maxiter = maxiter
        self.eps = eps
        self.ftol = ftol
        super(Geomed, self).__init__()
    
    def _geometric_median_objective(self, median, points, alphas):
        return sum([alpha * _compute_euclidean_distance(median, p) for alpha, p in zip(alphas, points)])
    
    def __call__(self, inputs: Union[List[BladesClient], List[torch.Tensor]], weights=None):
        updates = self._get_updates(inputs)
        if weights is None:
            weights = np.ones(len(updates)) / len(updates)
        median = updates.mean(dim=0)
        num_oracle_calls = 1
        obj_val = self._geometric_median_objective(median, updates, weights)
        for i in range(self.maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray(
                [max(self.eps, alpha / max(self.eps, _compute_euclidean_distance(median, p).item())) for alpha, p in
                 zip(weights, updates)],
                dtype=weights.dtype)
            weights = weights / weights.sum()
            median = torch.sum(torch.vstack([w * beta for w, beta in zip(updates, weights)]), dim=0)
            num_oracle_calls += 1
            obj_val = self._geometric_median_objective(median, updates, weights)
            if abs(prev_obj_val - obj_val) < self.ftol * obj_val:
                break
        
        return median
