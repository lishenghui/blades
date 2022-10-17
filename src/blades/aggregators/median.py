from typing import List, Union

import torch

from blades.clients.client import BladesClient
from .mean import _BaseAggregator


class Median(_BaseAggregator):
    r"""A robust aggregator from paper `Byzantine-robust distributed learning:
    Towards optimal statistical rates.

    <https://proceedings.mlr.press/v80/yin18a>`_.

    It computes the coordinate-wise median of the given set of clients
    """

    def __int__(self):
        super(Median, self).__init__()

    def __call__(
        self,
        inputs: Union[
            List[BladesClient],
            List[torch.Tensor],
            torch.Tensor,
        ],
    ):
        updates = self._get_updates(inputs)
        values_upper, _ = updates.median(dim=0)
        values_lower, _ = (-updates).median(dim=0)
        return (values_upper - values_lower) / 2
