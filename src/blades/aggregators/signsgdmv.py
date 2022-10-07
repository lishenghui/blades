from typing import List, Union

import torch

from blades.clients.client import BladesClient
from .mean import _BaseAggregator


class Signsgdmv(_BaseAggregator):
    r"""A robust aggregator from paper `signSGD with Majority Vote is
    Communication Efficient And Fault Tolerant`__.

    __ <https://arxiv.org/abs/1810.05291
    """

    def __int__(self):
        super(Signsgdmv, self).__init__()

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
