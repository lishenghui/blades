import copy
import inspect
import os
import sys
from collections import defaultdict
from typing import Union, Callable, Tuple

import ray
import ray.train as train
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from torch.nn.modules.loss import CrossEntropyLoss


class TorchClient(object):
    def __init__(
            self,
            client_id, train_data, test_data,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_func: torch.nn.modules.loss._Loss,
            device: Union[torch.device, str],
    ):
        self.id = client_id
        self.raw_train_data = train_data
        self.raw_test_data = test_data
        self.model = model.to('cpu')
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        
        self.__is_byzantine = False
        self.running = {}
        self.metrics = {}
        self.state = defaultdict(dict)
    
    def detach_model(self):
        self.model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        self.loss_func = CrossEntropyLoss().to(self.device)
    
    def getattr(self, attr):
        return getattr(self, attr)
    
    def get_is_byzantine(self):
        return self.__is_byzantine
    
    def omniscient_callback(self, simulator):
        pass
    
    def set_para(self, model):
        self.model.load_state_dict(model.state_dict())  # .to(self.device)
    
    def add_metric(self, name: str, callback: Callable[[torch.Tensor, torch.Tensor], float]):
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")
        
        self.metrics[name] = callback
    
    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])
    
    def __str__(self) -> str:
        return "TorchWorker"
    
    def train_epoch_start(self) -> None:
        # self.running["train_loader_iterator"] = iter(self.data_loader)
        self.model = self.model.to(self.device)
        self.model.train()
    
    def reset_data_loader(self):
        self.running["train_loader_iterator"] = iter(self.data_loader)
    
    def compute_gradient(self) -> Tuple[float, int]:
        results = {}
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
        self._save_grad()
        
        self.running["data"] = data
        self.running["target"] = target
        
        results["loss"] = loss.item()
        results["length"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)
        return results
    
    # def local_training(self, num_rounds) -> Tuple[float, int]:
    #     self._save_para()
    #     results = {}
    #     for _ in range(num_rounds):
    #         try:
    #             data, target = self.running["train_loader_iterator"].__next__()
    #         except StopIteration:
    #             self.reset_data_loader()
    #             data, target = self.running["train_loader_iterator"].__next__()
    #         data, target = data.to(self.device), target.to(self.device)
    #         self.optimizer.zero_grad()
    #         output = self.model(data)
    #         loss = self.loss_func(output, target)
    #         loss.backward()
    #         self.apply_gradient()
    #
    #     self._save_update()
    #
    #     self.running["data"] = data
    #     self.running["target"] = target
    #
    #     results["loss"] = loss.item()
    #     results["length"] = len(target)
    #     results["metrics"] = {}
    #     for name, metric in self.metrics.items():
    #         results["metrics"][name] = metric(output, target)
    #     return results
    
    def local_training(self, num_rounds, data_batches) -> Tuple[float, int]:
        self._save_para()
        results = {}
        model = train.torch.prepare_model(self.model)
        model = self.model
        # for _ in range(num_rounds):
        for data, target in data_batches:
            # try:
            #     data, target = self.running["train_loader_iterator"].__next__()
            # except StopIteration:
            #     self.reset_data_loader()
            #     data, target = self.running["train_loader_iterator"].__next__()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output = model(data)
            # output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.apply_gradient()
        
        self.model = model
        self._save_update()
        
        self.running["data"] = data
        self.running["target"] = target
        
        results["loss"] = loss.item()
        results["length"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)
        return results
    
    def get_gradient(self) -> torch.Tensor:
        return self._get_saved_grad()
    
    def get_update(self) -> torch.Tensor:
        return torch.nan_to_num(self._get_saved_update())
    
    def apply_gradient(self) -> None:
        self.optimizer.step()
    
    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end
    
    # def set_para(self, para: torch.Tensor) -> None:
    #     beg = 0
    #     for p in self.model.parameters():
    #         end = beg + len(p.grad.view(-1))
    #         x = para[beg:end].reshape_as(p.data)
    #         p.data = x.clone().detach()
    #         beg = end
    
    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state["saved_grad"] = torch.clone(p.grad).detach()
    
    def _save_update(self) -> None:
        self.state['saved_update'] = (self._get_para(current=True) - self._get_para(current=False)).detach()
    
    def _get_saved_update(self):
        return self.state['saved_update']
    
    def _save_para(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                param_state = self.state[p]
                param_state["saved_para"] = torch.clone(p.data).detach()
    
    def _get_para(self, current=True) -> None:
        layer_parameters = []
        
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if current:
                    layer_parameters.append(p.data.view(-1))
                else:
                    param_state = self.state[p]
                    layer_parameters.append(param_state["saved_para"].data.view(-1))
        return torch.cat(layer_parameters).to('cpu')
    
    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["saved_grad"].data.view(-1))
        return torch.cat(layer_gradients)


@ray.remote
class RemoteWorker(TorchClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@ray.remote
class WorkerWithMomentum(TorchClient):
    """
    Note that we use `WorkerWithMomentum` instead of using multiple `torch.optim.Optimizer`
    because we need to explicitly update the `momentum_buffer`.
    """
    
    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
    
    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                else:
                    param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad.mul_(1 - self.momentum))
    
    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["momentum_buffer"].data.view(-1))
        return torch.cat(layer_gradients)


# @ray.remote
class ByzantineWorker(TorchClient):
    def __int__(self, *args, **kwargs):
        super(ByzantineWorker).__init__(*args, **kwargs)
    
    def configure(self, simulator):
        # call configure after defining DistribtuedSimulator
        self.simulator = simulator
        simulator.register_omniscient_callback(self.omniscient_callback)
    
    def compute_gradient(self) -> Tuple[float, int]:
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return super().compute_gradient()
    
    def get_gradient(self) -> torch.Tensor:
        # Use self.simulator to get all other workers
        return super().get_gradient()
    
    # def omniscient_callback(self):
    #     raise NotImplementedError
    
    def __str__(self) -> str:
        return "ByzantineWorker"
