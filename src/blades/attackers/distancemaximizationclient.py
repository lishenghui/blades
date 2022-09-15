import torch

from blades.client import ByzantineClient

class DistancemaximizationClient(ByzantineClient):
    def omniscient_callback(self, simulator):
        pass
    
    def local_training(self, data_batches):
        pass
    
class DistancemaximizationAdversary():
    r""" 
    """
    
    def __init__(self, num_byzantine: int, agg: str, dev_type='sign', threshold=5.0, threshold_diff=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = dev_type
        self.agg = agg
        self.threshold = threshold
        self.threshold_diff = threshold_diff
        self.num_byzantine = num_byzantine
    
    def attack(self, simulator):
        all_updates = torch.stack(list(map(lambda w: w.get_update(), simulator._clients.values())))
        model_re = torch.mean(all_updates, 0)
    
        if self.dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif self.dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)
        elif self.dev_type == 'std':
            deviation = torch.std(all_updates, 0)
    
        lamda = torch.Tensor([self.threshold]).to(all_updates.device)
    
        threshold_diff = self.threshold_diff
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0
    
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * self.num_byzantine)
            mal_updates = torch.cat((mal_updates, all_updates), 0)
        
            agg_updates = simulator.server.aggregator(mal_updates)
        
            loss = torch.norm(agg_updates - model_re)
        
            if prev_loss < loss:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2
        
            lamda_fail = lamda_fail / 2
            prev_loss = loss
    
        mal_update = (model_re - lamda_succ * deviation)
        for i, client in enumerate(simulator._clients.values()):
            if client.is_byzantine():
                client.save_update(mal_update)
    
    def omniscient_callback(self, simulator):
        self.attack(simulator)
