import torch

from .mean import _BaseAggregator


class Median(_BaseAggregator):
    r"""
    
      A robust aggregator from paper `Byzantine-robust distributed learning: Towards optimal statistical rates <https://proceedings.mlr.press/v80/yin18a>`_
      
      It computes the coordinate-wise median of the given set of updates
      
    """
    
    def __int__(self):
        super(Median, self).__init__()
    
    def __call__(self, clients):
        updates = list(map(lambda w: w.get_update(), clients))
        stacked = torch.stack(updates, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2
