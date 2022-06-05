import inspect
import os
import sys

import numpy as np
import ray
import torch
from scipy.stats import norm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from client import ByzantineWorker


# @ray.remote
class AlieClient(ByzantineWorker):
    """
    Args:
        n (int): Total number of workers
        m (int): Number of Byzantine workers
    """
    
    def __init__(self, n, m, is_fedavg=True, z=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__fedavg = is_fedavg
        # Number of supporters
        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
        self.n_good = n - m
        self.__is_byzantine = True
    
    def get_is_byzantine(self):
        return self.__is_byzantine
    
    def get_gradient(self):
        return self._gradient
    
    def omniscient_callback(self, simulator):
        # Loop over good workers and accumulate their gradients
        updates = []
        for w in simulator._clients:
            is_byzantine = w.get_is_byzantine()
            # is_byzantine = ray.get(w.getattr.remote('__is_byzantine'))
            if not is_byzantine:
                updates.append(w.get_update())
        
        stacked_updates = torch.stack(updates, 1)
        mu = torch.mean(stacked_updates, 1)
        std = torch.std(stacked_updates, 1)
        
        self._gradient = mu - std * self.z_max
        if self.__fedavg:
            self.state['saved_update'] = self._gradient
    
    # def set_gradient(self, gradient) -> None:
    #     raise NotImplementedError
    #
    # def apply_gradient(self) -> None:
    #     raise NotImplementedError
    
    # def local_training(self, num_rounds, use_actor, data_batches):
    #     pass
