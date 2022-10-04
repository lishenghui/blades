import torch

from blades.core.client import ByzantineClient


class SignflippingClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_global_model(self, data_batches, opt):
        for data, target in data_batches:
            data, target = data.to(self.device), target.to(self.device)
            data, target = self.on_train_batch_begin(data=data, target=target)
            opt.zero_grad()

            output = self.global_model(data)
            loss = torch.clamp(self.loss_func(output, target), 0, 1e5)
            loss.backward()
            for name, p in self.global_model.named_parameters():
                p.grad.data = -p.grad.data
            opt.optimizer.step()
