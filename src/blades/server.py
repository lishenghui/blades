import torch


class BladesServer(object):
    r'''Simulating the server of the federated learning system.
        
        :param  optimizer: The global optimizer, which can be any optimizer from Pytorch.
        :type optimizer: torch.optim.Optimizer
        :param model: The global model
        :type model: torch.nn.Module
    '''
    def __init__(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module):
        self.optimizer = optimizer
        self.model = model
    
    def get_opt(self) -> torch.optim.Optimizer:
        r'''
        Returns the global optimizer.
        '''
        return self.optimizer
    
    def get_model(self):
        r'''
        Returns the current global model.
        '''
        return self.model
    
    def apply_update(self, update: torch.Tensor) -> None:
        r'''
        Apply a stop of global optimization.
            
            .. note::
                The input should be a ``Tensor``, which will be converted to ``pseudo-gradient``
                layer by layer.
                
        :param update: The aggregated update.
        :type update: torch.Tensor
        '''
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                end = beg + len(p.data.view(-1))
                x = update[beg:end].reshape_as(p.data)
                p.grad.data = -x.clone().detach()
                beg = end
        self.optimizer.step()