import numpy as np
import torch

from blades.client import ByzantineClient


class FangattackClient(ByzantineClient):
    def omniscient_callback(self, simulator):
        pass
    
    def local_training(self, data_batches):
        pass
    
class FangattackAdversary():
    r""" 
    """
    
    def __init__(self, num_byzantine: int, agg: str, dev_type='std', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = dev_type
        self.agg = agg
        self.num_byzantine = num_byzantine
    
    def attack_median_and_trimmedmean(self, simulator):
        benign_update = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                benign_update.append(w.get_update())
        benign_update = torch.stack(benign_update, 0)
        agg_grads = torch.mean(benign_update, 0)
        deviation = torch.sign(agg_grads)
        b = 2
        max_vector = torch.max(benign_update, 0)[0]
        min_vector = torch.min(benign_update, 0)[0]
        
        max_ = (max_vector > 0).type(torch.FloatTensor)
        min_ = (min_vector < 0).type(torch.FloatTensor)
        
        max_[max_ == 1] = b
        max_[max_ == 0] = 1 / b
        min_[min_ == 1] = b
        min_[min_ == 0] = 1 / b
        
        max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
        min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)
        
        rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), self.num_byzantine])).type(torch.FloatTensor)
        
        max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
            [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
            [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
        
        mal_vec = (torch.stack(
            [(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]).T * max_rand + torch.stack(
            [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T * min_rand).T
        
        for i, client in enumerate(simulator._clients.values()):
            if client.is_byzantine():
                client.save_update(mal_vec[i])
    
    def omniscient_callback(self, simulator):
        if self.agg in ["median", "trimmedmean"]:
            self.attack_median_and_trimmedmean(simulator)

