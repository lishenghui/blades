from typing import Optional

import torch

from blades.client import ByzantineClient


class NoiseClient(ByzantineClient):
    r"""Uploads random noise as local update. The noise is drawn from a
    ``normal`` distribution.  The ``means`` and ``standard deviation`` are shared among all drawn elements.
    
    :param mean: the mean for all distributions
    :param std: the standard deviation for all distributions
    """
    
    def __init__(self, mean: Optional[float] = 0.1, std: Optional[float] = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_mean = mean
        self._noise_std = std
    
    def omniscient_callback(self, simulator):
        self._state['saved_update'] = torch.normal(self._noise_mean,
                                                   self._noise_std,
                                                   size=super().get_update().shape
                                                   ).to('cpu')
