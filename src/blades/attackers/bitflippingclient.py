import inspect
import os
import sys

import ray

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from simulator.client import ByzantineWorker


@ray.remote
class BitflippingClient(ByzantineWorker):
    def __str__(self) -> str:
        return "BitFlippingWorker"
    
    def get_gradient(self):
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return -super().get_gradient()
    
    def local_training(self, num_rounds):
        self._save_para()
        results = {}
        for _ in range(num_rounds):
            try:
                data, target = self.running["train_loader_iterator"].__next__()
            except StopIteration:
                self.reset_data_loader()
                data, target = self.running["train_loader_iterator"].__next__()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
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
