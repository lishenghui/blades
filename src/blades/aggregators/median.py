import torch

from .mean import _BaseAggregator


class Median(_BaseAggregator):
    
    def __int__(self):
        super(Median, self).__init__()
    
    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2