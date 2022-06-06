import inspect
import os
import sys
import ray.train as train
import ray

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from client import ByzantineWorker


class SignflippingClient(ByzantineWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def get_gradient(self):
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return -super().get_gradient()
    
    def local_training(self, num_rounds, use_actor, data_batches):
        self._save_para()
        results = {}

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
            self.apply_gradient()

        self._save_update()
        
        self.running["data"] = data
        self.running["target"] = target
        
        results["loss"] = loss.item()
        results["length"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)
        return results
