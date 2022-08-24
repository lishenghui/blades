from blades.client import ByzantineClient


class IpmClient(ByzantineClient):
    def __init__(self, epsilon: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        
    def omniscient_callback(self, simulator):
        # Loop over good workers and accumulate their updates
        updates = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                updates.append(w.get_update())
        
        self.save_update(-self.epsilon * (sum(updates)) / len(updates))