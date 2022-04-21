"""
A better name will be Inner Product Manipulation Attack.
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
import torch
from simulators.worker import ByzantineWorker


class NoiseAttack(ByzantineWorker):
    def __init__(self, is_fedavg, noise=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__fedavg = is_fedavg
        self.__noise = noise
        self._gradient = None
    
    def get_gradient(self):
        noise = 0.1
        # return torch.ones_like(super().get_gradient()).to(self.device) / 3
        return torch.normal(self.__noise, self.__noise, size=super().get_gradient().shape).to(self.device)
        # return self._gradient
    
    def omniscient_callback(self):
        if self.__fedavg:
            self.state['saved_update'] = torch.normal(self.__noise, self.__noise, size=super().get_update().shape).to(
                self.device)
    
    # def set_gradient(self, gradient) -> None:
    #     raise NotImplementedError
    
    # def apply_gradient(self) -> None:
    #     raise NotImplementedError
    
    # def local_training(self, num_rounds):
    #     self._save_para()
    #     results = {}
    #     data, target = self.running["train_loader_iterator"].__next__()
    #     data, target = data.to(self.device), target.to(self.device)
    #     output = self.model(data)
    #     loss = self.loss_func(output, target)
    #     loss.backward()
    #     self._save_update()
    
    #     self.running["data"] = data
    #     self.running["target"] = target
    
    #     results["loss"] = loss.item()
    #     results["length"] = len(target)
    #     results["metrics"] = {}
    #     for name, metric in self.metrics.items():
    #         results["metrics"][name] = metric(output, target)
    #     return results
