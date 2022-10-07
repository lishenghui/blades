from typing import List, Union

import torch

from blades.clients.client import BladesClient
from .mean import _BaseAggregator


def _compute_scores(distances, i, n, f):
    """Compute scores for node i.

    Args:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
        i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.

    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)


def _multi_krum(distances, n, f, m):
    """Multi_Krum algorithm.

    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
         i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.

    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: "
                    "Got {}.".format(i, j, distances[i][j])
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def _pairwise_euclidean_distances(vectors):
    """Compute the pairwise euclidean distance.

    Arguments:
        vectors {list} -- A list of vectors.

    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    n = len(vectors)
    vectors = [v.flatten() for v in vectors]

    distances = {}
    for i in range(n - 1):
        distances[i] = {}
        for j in range(i + 1, n):
            distances[i][j] = _compute_euclidean_distance(vectors[i], vectors[j]) ** 2
    return distances


class Multikrum(_BaseAggregator):
    r"""A robust aggregator from paper `Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent.

    <https://dl.acm.org/doi/abs/10.5555/3294771.3294783>`_.

    Given a collection of vectors, ``Krum`` strives to find one of the vector that is
    closest to another :math:`K-M-2` ones with respect to squared Euclidean distance,
    which can be expressed by:

      .. math::
         Krum := \{{\Delta}_i | i = \arg\min_{i \in [K]} \sum_{i \rightarrow j}  \lVert
         {\Delta}_i - {\Delta}_j \rVert^2 \}

    where :math:`i \rightarrow j` is the indices of the :math:`K-M-2` nearest neighbours
    of :math:`{\Delta}_i` measured by squared ``Euclidean distance``,  :math:`K` is the
    number of input in total, and :math:`M` is the number of Byzantine input.

    Args:
          lr (float): target learning rate.
    """

    def __init__(self, num_excluded=5, k=1):
        self.f = num_excluded
        self.m = k
        super(Multikrum, self).__init__()

    def __call__(self, inputs: Union[List[BladesClient], List[torch.Tensor]]):
        updates = self._get_updates(inputs)
        distances = _pairwise_euclidean_distances(updates)
        top_m_indices = _multi_krum(distances, len(updates), self.f, self.m)
        values = torch.stack([updates[i] for i in top_m_indices], dim=0).mean(dim=0)
        return values

    def __str__(self):
        return "Krum (m={})".format(self.m)
