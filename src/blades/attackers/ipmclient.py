import os
import sys
from pathlib import Path

import ray

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

from client import ByzantineWorker


class IpmClient(ByzantineWorker):
    def __init__(self, epsilon: float = 0.5, is_fedavg: bool = True, *args, **kwargs):
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
        updates = []
        for w in simulator._clients:
            is_byzantine = w.get_is_byzantine()
            # is_byzantine = ray.get(w.getattr.remote('__is_byzantine'))
            if not is_byzantine:
                updates.append(w.get_update())
        
        self._gradient = -self.epsilon * (sum(updates)) / len(updates)
        if self.__fedavg:
            self.state['saved_update'] = self._gradient