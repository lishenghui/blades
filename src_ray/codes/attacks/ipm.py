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
class IPMAttack(ByzantineWorker):
    def __init__(self, epsilon, is_fedavg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.__fedavg = is_fedavg
        self._gradient = None
    
    def get_gradient(self):
        return self._gradient
    
    def omniscient_callback(self):
        # Loop over good workers and accumulate their gradients
        update = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                if self.__fedavg:
                    update.append(w.get_update())
                else:
                    update.append(w.get_gradient())
        
        self._gradient = -self.epsilon * (sum(update)) / len(update)
        if self.__fedavg:
            self.state['saved_update'] = self._gradient
    
    def set_gradient(self, gradient) -> None:
        raise NotImplementedError
    
    def apply_gradient(self) -> None:
        raise NotImplementedError
    
    def local_training(self, num_rounds):
        pass
