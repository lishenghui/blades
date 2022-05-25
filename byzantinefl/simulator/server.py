import torch


class TorchServer(object):
    def __init__(self, optimizer: torch.optim.Optimizer, model):
        self.optimizer = optimizer
        self.model = model
    
    
    def get_opt(self):
        return self.optimizer
    
    def get_model(self):
        return self.model
    
    def apply_gradient(self) -> None:
        self.optimizer.step()
    
    def apply_update(self, update: torch.Tensor) -> None:
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                # for p in self.model.parameters():
                end = beg + len(p.data.view(-1))
                x = update[beg:end].reshape_as(p.data)
                p.data += x.clone().detach()
                beg = end
    
    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # for p in self.model.parameters():
                end = beg + len(p.grad.view(-1))
                x = gradient[beg:end].reshape_as(p.grad.data)
                p.grad.data = x.clone().detach()
                beg = end
