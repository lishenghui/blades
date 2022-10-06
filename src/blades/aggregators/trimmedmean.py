from typing import List, Union

import torch

from blades.clients.client import BladesClient
from .mean import _BaseAggregator


class Trimmedmean(_BaseAggregator):
    r"""A robust aggregator from paper `Byzantine-robust distributed learning:
    Towards optimal statistical rates.

    <https://proceedings.mlr.press/v80/yin18a>`_

    It computes the coordinate-wise trimmed average of the global_model updates,
    which can be expressed by:

    .. math::
             trmean := TrimmedMean( \{ \Delta_k : k \in [K] \} ),

    where the :math:`i`-th coordinate :math:`trmean_i = \frac{1}{(1-2\beta)m}
    \sum_{x \in U_k}x`, and :math:`U_k`
    is a subset obtained by removing the largest and smallest :math:`\beta` fraction
    of its elements.
    """

    def __init__(self, num_excluded=5):
        self.b = num_excluded
        super(Trimmedmean, self).__init__()

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
        if len(updates) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(updates) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError

        largest, _ = torch.topk(updates, b, 0)
        neg_smallest, _ = torch.topk(-updates, b, 0)
        new_stacked = torch.cat([updates, -largest, neg_smallest]).sum(0)
        new_stacked /= len(updates) - 2 * b
        return new_stacked

    def __str__(self):
        return "Trimmed Mean (b={})".format(self.b)
