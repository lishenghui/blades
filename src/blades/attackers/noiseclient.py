import os
import sys
from pathlib import Path

import ray

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import torch
from client import ByzantineWorker


class NoiseClient(ByzantineWorker):
    def __init__(self, is_fedavg=True, noise=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__fedavg = is_fedavg
        self.__noise = noise
        self._gradient = None
    
    def get_gradient(self):
        return torch.normal(self.__noise, self.__noise, size=super().get_gradient().shape).to(self.device)
    
    def omniscient_callback(self, simulator):
        if self.__fedavg:
            self.state['saved_update'] = torch.normal(self.__noise, self.__noise, size=super().get_update().shape).to(
                'cpu')
