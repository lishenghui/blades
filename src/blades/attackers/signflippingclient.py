import torch

from blades.client import ByzantineClient


class SignflippingClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def local_training(self, data_batches):
        for data, target in data_batches:
            data, target = data.to(self.device), target.to(self.device)
            data, target = self.on_train_batch_begin(data=data, target=target)
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = torch.clamp(self.loss_func(output, target), 0, 1e5)
            loss.backward()
            for name, p in self.model.named_parameters():
                p.grad.data = -p.grad.data
            self.optimizer.step()