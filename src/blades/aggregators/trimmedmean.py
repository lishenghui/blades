import torch

from .mean import _BaseAggregator


class Trimmedmean(_BaseAggregator):
    def __init__(self, num_byzantine=5):
        self.b = num_byzantine
        super(Trimmedmean, self).__init__()
    
    def __call__(self, clients):
        updates = list(map(lambda w: w.get_update(), clients))
        if len(updates) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(updates) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError
        
        stacked = torch.stack(updates, dim=0)
        largest, _ = torch.topk(stacked, b, 0)
        neg_smallest, _ = torch.topk(-stacked, b, 0)
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= len(updates) - 2 * b
        return new_stacked
    
    def __str__(self):
        return "Trimmed Mean (b={})".format(self.b)
