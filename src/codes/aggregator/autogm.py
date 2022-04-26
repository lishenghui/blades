import numpy as np
import torch

from .base import _BaseAggregator
from .rfa import RFA


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


class AutoGM(_BaseAggregator):
    def __init__(self):
        super(AutoGM, self).__init__()
        self.gm_agg = RFA()
        self.momentum = None
    
    def geometric_median_objective(self, median, points, alphas):
        return sum([alpha * _compute_euclidean_distance(median, p) for alpha, p in zip(alphas, points)])
    
    def __call__(self, inputs, weights=None, maxiter=100, eps=1e-6, ftol=1e-6):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
        
        lamb = 1 * len(inputs)
        alpha = np.ones(len(inputs)) / len(inputs)
        median = self.gm_agg(inputs, alpha)
        obj_val = self.geometric_median_objective(median, inputs, alpha)
        global_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
        distance = np.zeros_like(alpha)
        for i in range(maxiter):
            prev_global_obj = global_obj
            for idx, local_model in enumerate(inputs):
                distance[idx] = _compute_euclidean_distance(local_model, median)
            
            idxs = [x for x, _ in sorted(enumerate(distance), key=lambda x: x)]
            eta_optimal = 10000000000000000.0
            for p in range(0, len(idxs)):
                eta = (sum([distance[i] for i in idxs[:p + 1]]) + lamb) / (p + 1)
                if p < len(idxs) and eta - distance[idxs[p]] < 0:
                    break
                else:
                    eta_optimal = eta
            alpha = np.array([max(eta_optimal - d, 0) / lamb for d in distance])
            
            median = self.gm_agg(inputs, alpha)
            gm_sum = self.geometric_median_objective(median, inputs, alpha)
            global_obj = gm_sum + lamb * np.linalg.norm(alpha) ** 2 / 2
            if abs(prev_global_obj - global_obj) < ftol * global_obj:
                break
        self.momentum = median
        return self.momentum
