from blades.client import ByzantineClient


class IpmClient(ByzantineClient):
    def __init__(self, epsilon: float = 0.5, is_fedavg: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.__fedavg = is_fedavg
        self._gradient = None
        self._is_byzantine = True
    
    def is_byzantine(self):
        return self._is_byzantine
    
    def get_gradient(self):
        return self._gradient
    
    def omniscient_callback(self, simulator):
        # Loop over good workers and accumulate their gradients
        updates = []
        for w in simulator.get_clients():
            is_byzantine = w.is_byzantine()
            # is_byzantine = ray.get(w.getattr.remote('_is_byzantine'))
            if not is_byzantine:
                updates.append(w.get_update())
        
        self._gradient = -self.epsilon * (sum(updates)) / len(updates)
        if self.__fedavg:
            self._state['saved_update'] = self._gradient
