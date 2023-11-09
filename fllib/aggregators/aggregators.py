from typing import List, Optional

import numpy as np
import torch


def _mean(inputs: List[torch.Tensor]):
    inputs_tensor = torch.stack(inputs, dim=0)
    return inputs_tensor.mean(dim=0)


def _median(inputs: List[torch.Tensor]):
    inputs_tensor = torch.stack(inputs, dim=0)
    values_upper, _ = inputs_tensor.median(dim=0)
    values_lower, _ = (-inputs_tensor).median(dim=0)
    return (values_upper - values_lower) / 2


class Mean(object):
    def __call__(self, inputs: List[torch.Tensor]):
        return _mean(inputs)


class Median(object):
    def __call__(self, inputs: List[torch.Tensor]):
        return _median(inputs)


class Trimmedmean(object):
    def __init__(self, num_byzantine: int, *, filter_frac=1.0):
        if filter_frac > 1.0 or filter_frac < 0.0:
            raise ValueError(f"filter_frac should be in [0.0, 1.0], got {filter_frac}.")

        if not isinstance(num_byzantine, int):
            raise ValueError(
                f"num_byzantine should be an integer, got {num_byzantine}."
            )

        def round_up_to_power_of_two(num):
            num = int(num)
            return num

        self.num_excluded = round_up_to_power_of_two(filter_frac * num_byzantine)

    def __call__(self, inputs: List[torch.Tensor]):
        if len(inputs) <= self.num_excluded * 2:
            raise ValueError(
                f"Not enough inputs to compute trimmed mean,"
                f"got {len(inputs)} inputs but need at least "
                f"{self.num_excluded * 2 + 1} inputs."
            )
        inputs = torch.stack(inputs, dim=0)
        largest, _ = torch.topk(inputs, self.num_excluded, 0)
        neg_smallest, _ = torch.topk(-inputs, self.num_excluded, 0)
        new_stacked = torch.cat([inputs, -largest, neg_smallest]).sum(0)
        new_stacked /= len(inputs) - 2 * self.num_excluded
        return new_stacked


class GeoMed:
    def __init__(
        self,
        maxiter: Optional[int] = 100,
        eps: Optional[float] = 1e-6,
        ftol: Optional[float] = 1e-10,
    ):
        self.maxiter = maxiter
        self.eps = eps
        self.ftol = ftol

    def __call__(self, inputs: List[torch.Tensor], weights=None):
        if weights is None:
            weights = (torch.ones(len(inputs)) / len(inputs)).to(inputs[0].device)
        input_tensor = torch.stack(inputs, dim=0)
        return self._geometric_median(
            input_tensor,
            weights=weights,
            maxiter=self.maxiter,
            eps=self.eps,
            ftol=self.ftol,
        )

    @staticmethod
    def _geometric_median(inputs, weights, eps=1e-6, maxiter=100, ftol=1e-20):
        weighted_average = (
            lambda inputs, weights: torch.sum(weights.view(-1, 1) * inputs, dim=0)
            / weights.sum()
        )

        def obj_func(median, inputs, weights):
            # This function is not used so far,
            # as the numpy version appears to be more accurate (and I don't know why).

            # norms = torch.norm(inputs - median, dim=1)
            # return (torch.sum(norms * weights) / torch.sum(weights)).item()

            return np.average(
                [torch.norm(p - median).item() for p in inputs],
                weights=weights.cpu(),
            )

        with torch.no_grad():
            median = weighted_average(inputs, weights)
            new_weights = weights
            objective_value = obj_func(median, inputs, weights)

            # Weiszfeld iterations
            for _ in range(maxiter):
                prev_obj_value = objective_value
                denom = torch.stack([torch.norm(p - median) for p in inputs])
                new_weights = weights / torch.clamp(denom, min=eps)
                median = weighted_average(inputs, new_weights)

                objective_value = obj_func(median, inputs, weights)
                if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                    break

        median = weighted_average(inputs, new_weights)
        return median


class DnC(object):
    r"""A robust aggregator from paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.

    <https://par.nsf.gov/servlets/purl/10286354>`_.
    """

    def __init__(
        self, num_byzantine, *, sub_dim=10000, num_iters=5, filter_frac=1.0
    ) -> None:
        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.fliter_frac = filter_frac

    def __call__(self, inputs: List[torch.Tensor]):
        updates = torch.stack(inputs, dim=0)
        d = len(updates[0])

        benign_ids = []
        for i in range(self.num_iters):
            indices = torch.randperm(d)[: self.sub_dim]
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
            s = np.array(
                [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
            )

            good = s.argsort()[
                : len(updates) - int(self.fliter_frac * self.num_byzantine)
            ]
            benign_ids.append(good)

        # Convert the first list to a set to start the intersection
        intersection_set = set(benign_ids[0])

        # Iterate over the rest of the lists and get the intersection
        for lst in benign_ids[1:]:
            intersection_set.intersection_update(lst)

        # Convert the set back to a list
        benign_ids = list(intersection_set)
        benign_updates = updates[benign_ids, :].mean(dim=0)
        return benign_updates
