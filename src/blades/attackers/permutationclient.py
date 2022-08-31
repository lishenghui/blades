import torch

from blades.client import ByzantineClient


class PermutationClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def omniscient_callback(self, simulator):
        # Loop over good workers and accumulate their updates
        true_update = self.get_update()
        r = torch.randperm(len(true_update))
        malicious_update = true_update[r]
        self.save_update(malicious_update)