import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import norm

from ..simulators.worker import ByzantineWorker


class ALittleIsEnoughAttack(ByzantineWorker):
    """
    Args:
        n (int): Total number of workers
        m (int): Number of Byzantine workers
    """
    
    def __init__(self, n, m, is_fedavg, z=None, *args, **kwargs):
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
    
    def get_gradient(self):
        return self._gradient
    
    def omniscient_callback(self):
        # Loop over good workers and accumulate their gradients
        updates = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                if self.__fedavg:
                    updates.append(w.get_update())
                else:
                    updates.append(w.get_gradient())
        
        stacked_updates = torch.stack(updates, 1)
        mu = torch.mean(stacked_updates, 1)
        std = torch.std(stacked_updates, 1)
        
        self._gradient = mu - std * self.z_max
        if self.__fedavg:
            self.state['saved_update'] = self._gradient
    
    def set_gradient(self, gradient) -> None:
        raise NotImplementedError
    
    def apply_gradient(self) -> None:
        raise NotImplementedError
    
    def local_training(self, num_rounds):
        pass
