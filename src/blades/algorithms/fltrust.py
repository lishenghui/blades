import torch
from blades.server import BladesServer

class _FltrustedAGG(object):
    
    def _get_trusted
    def __call__(self, updates, clients):
        stacked = torch.stack(updates, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2


class FltrustServer(BladesServer):
    def __int__(self, *args, **kwargs):
        super(FltrustServer, self).__int__(*args, **kwargs)
    
    
