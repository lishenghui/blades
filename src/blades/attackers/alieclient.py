import numpy as np
import torch
from scipy.stats import norm

from blades.client import ByzantineClient


class AlieClient(ByzantineClient):
    """
    Args:
        n (int): Total number of workers
        m (int): Number of Byzantine workers
    """
    
    def __init__(self, n, m, z=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Number of supporters
        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
        self.n_good = n - m
    
    def get_gradient(self):
        return self._gradient
    
    def omniscient_callback(self, simulator):
        # Loop over good workers and accumulate their gradients
        updates = []
        for client in simulator._clients:
            if not client.get_is_byzantine():
                updates.append(client.get_update())
        
        stacked_updates = torch.stack(updates, 1)
        mu = torch.mean(stacked_updates, 1)
        std = torch.std(stacked_updates, 1)
        
        self._gradient = mu - std * self.z_max
        self.state['saved_update'] = self._gradient
