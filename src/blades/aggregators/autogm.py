from typing import List, Optional, Union

import numpy as np
import torch

from blades.clients.client import BladesClient
from .geomed import Geomed
from .mean import _BaseAggregator


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


class Autogm(_BaseAggregator):
    r"""A robust aggregator from paper `Byzantine-Robust Aggregation in
    Federated Learning Empowered Industrial IoT.

    <https://ieeexplore.ieee.org/abstract/document/9614992>`_.

    Args:
        maxiter: Maximum number of Weiszfeld iterations, default 2.0.
        eps: Smallest allowed value of denominator, to avoid divide by zero.
        ftol: If objective value does not improve by at least this `ftol` fraction,
                terminate the algorithm. Default 1e-10.
    """

    def __init__(
        self,
        lamb: Optional[float] = 2.0,
        maxiter: Optional[int] = 100,
        eps: Optional[float] = 1e-6,
        ftol: Optional[float] = 1e-10,
    ):
        super(Autogm, self).__init__()
        self.lamb = lamb
        self.maxiter = maxiter
        self.eps = eps
        self.ftol = ftol
        self.gm_agg = Geomed(maxiter=maxiter, eps=eps, ftol=ftol)

    def geometric_median_objective(self, median, points, alphas):
        return sum(
            [
                alpha * _compute_euclidean_distance(median, p)
                for alpha, p in zip(alphas, points)
            ]
        )

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor]], weights=None
    ):
        updates = self._get_updates(inputs)

        lamb = 1 * len(updates) if self.lamb is None else self.lamb
        alpha = np.ones(len(updates)) / len(updates)
        median = self.gm_agg(updates, alpha)
        obj_val = self.geometric_median_objective(median, updates, alpha)
        global_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
        distance = np.zeros_like(alpha)
        for i in range(self.maxiter):
            prev_global_obj = global_obj
            for idx, local_model in enumerate(updates):
                distance[idx] = _compute_euclidean_distance(local_model, median)

            idxs = [x for x, _ in sorted(enumerate(distance), key=lambda x: x)]
            eta_optimal = 10000000000000000.0
            for p in range(0, len(idxs)):
                eta = (sum([distance[i] for i in idxs[: p + 1]]) + lamb) / (p + 1)
                if p < len(idxs) and eta - distance[idxs[p]] < 0:
                    break
                else:
                    eta_optimal = eta
            alpha = np.array([max(eta_optimal - d, 0) / lamb for d in distance])

            median = self.gm_agg(updates, alpha)
            gm_sum = self.geometric_median_objective(median, updates, alpha)
            global_obj = gm_sum + lamb * np.linalg.norm(alpha) ** 2 / 2
            if abs(prev_global_obj - global_obj) < self.ftol * global_obj:
                break
        return median
