from typing import List, Union

import numpy as np
import torch

from blades.clients.client import BladesClient
from .mean import _BaseAggregator


class Dnc(_BaseAggregator):
    r"""A robust aggregator from paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.

    <https://par.nsf.gov/servlets/purl/10286354>`_.
    """

    def __init__(
        self, num_byzantine, *, sub_dim=10000, num_iters=1, filter_frac=1.0
    ) -> None:
        super(Dnc, self).__init__()

        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.fliter_frac = filter_frac

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
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
            benign_ids.extend(good)

        benign_ids = list(set(benign_ids))
        benign_updates = updates[benign_ids, :].mean(dim=0)
        return benign_updates
