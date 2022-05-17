"""
A better name will be Inner Product Manipulation Attack.
"""
import os
import sys
from pathlib import Path

import ray

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

from simulators.worker import ByzantineWorker


@ray.remote
class IpmClient(ByzantineWorker):
    def __init__(self, epsilon, is_fedavg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.__fedavg = is_fedavg
        self._gradient = None
        self.__is_byzantine = True
    
    
    def get_is_byzantine(self):
        return self.__is_byzantine
    
    def get_gradient(self):
        return self._gradient
    
    def omniscient_callback(self, simulator):
        # Loop over good workers and accumulate their gradients
        update = []
        for w in simulator.workers:
            is_byzantine = ray.get(w.get_is_byzantine.remote())
            # is_byzantine = ray.get(w.getattr.remote('__is_byzantine'))
            if not is_byzantine:
                if self.__fedavg:
                    update.append(ray.get(w.get_update.remote()))
                else:
                    update.append(ray.get(w.get_gradient.remote()))
        
        self._gradient = -self.epsilon * (sum(update)) / len(update)
        if self.__fedavg:
            self.state['saved_update'] = self._gradient
    
    def set_gradient(self, gradient) -> None:
        raise NotImplementedError
    
    def apply_gradient(self) -> None:
        raise NotImplementedError
    
    def local_training(self, num_rounds):
        pass
