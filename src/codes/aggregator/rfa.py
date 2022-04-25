import numpy as np
import torch

from .base import _BaseAggregator


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


class RFA_back(_BaseAggregator):
    r""""""
    
    def __init__(self, T, nu=0.1):
        self.T = T
        self.nu = nu
        super(RFA, self).__init__()
    
    def __call__(self, inputs):
        alphas = [1 / len(inputs) for _ in inputs]
        z = torch.zeros_like(inputs[0])
        return smoothed_weiszfeld(inputs, alphas, z=z, nu=self.nu, T=self.T)
    
    def __str__(self):
        return "RFA(T={},nu={})".format(self.T, self.nu)


class RFA(_BaseAggregator):
    
    def geometric_median_objective(self, median, points, alphas):
        return sum([alpha * _compute_euclidean_distance(median, p) for alpha, p in zip(alphas, points)])
    
    def __call__(self, inputs, weights=None, maxiter=100, eps=1e-6, ftol=1e-6):
        if weights is None:
            weights = np.ones(len(inputs)) / len(inputs)
        median = torch.stack(inputs, dim=0).mean(dim=0)
        num_oracle_calls = 1
        obj_val = self.geometric_median_objective(median, inputs, weights)
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray(
                [max(eps, alpha / max(eps, _compute_euclidean_distance(median, p).item())) for alpha, p in
                 zip(weights, inputs)],
                dtype=weights.dtype)
            weights = weights / weights.sum()
            median = torch.sum(torch.vstack([w * beta for w, beta in zip(inputs, weights)]), dim=0)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, inputs, weights)
            # print('gm obj:', obj_val)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        
        return median
