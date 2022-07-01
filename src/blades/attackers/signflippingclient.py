import ray.train as train

from blades.client import ByzantineClient


class SignflippingClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def local_training(self, num_rounds, use_actor, data_batches):
        self._save_para()
        
        if use_actor:
            model = self.model
        else:
            model = train.torch.prepare_model(self.model)
        
        for data, target in data_batches:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            for name, p in self.model.named_parameters():
                p.grad.data = -p.grad.data
            self.optimizer.step()

        update = (self._get_para(current=True) - self._get_para(current=False))
        self.save_update(update)
