import copy
import logging
from collections import defaultdict
from typing import Union, Callable, Tuple, Optional

import ray.train as train
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class BladesClient(object):
    r"""Base class for all clients.
    
        .. note::
            Your honest clients should also subclass this class.
            
        :param client_id: a unique id of the client.
        :type client_id: str, optional.
        :param device:  if specified, all parameters will be copied to that device
        :type device: torch.device, or str
    """
    
    _is_byzantine: bool = False
    
    
    def __init__(
            self,
            client_id: Optional[str]=None,
            device: Union[torch.device, str]='cpu',
    ):
        self.id = client_id
        self.device = device
        
        self.running = {}
        self.state = defaultdict(dict)
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
    
    
    def set_id(self, id):
        self.id = id
        
    def get_id(self):
        return self.id
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def get_is_byzantine(self):
        return self._is_byzantine
    
    def set_model(self, model: nn.Module, opt: type(torch.optim.Optimizer), lr) -> None:
        r''' Deep copy the given model to the client.
        
            .. note::
                To improve the scalability, this API may be removed in the future,
                
            :param model: a Torch model for current client.
            :type model: torch.nn.Module
            :param opt: client optimizer
            :type opt: torch.optim.Optimizer
            :param lr:  local learning rate
            :type lr: float
        '''
        self.model = copy.deepcopy(model)
        self.optimizer = opt(self.model.parameters(), lr=lr)
    
    def set_loss(self, loss_func='crossentropy'):
        if loss_func == 'crossentropy':
            self.loss_func = nn.modules.loss.CrossEntropyLoss()
        else:
            raise NotImplementedError
    
    def set_para(self, model):
        self.model.load_state_dict(model.state_dict())  # .to(self.device)
    
    # def add_metric(self, name: str, callback: Callable[[torch.Tensor, torch.Tensor], float]):
    #     if name in self.metrics or name in ["loss", "length"]:
    #         raise KeyError(f"Metrics ({name}) already added.")
    #
    #     self.metrics[name] = callback
    
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
    
    def evaluate(self, round_number, test_set, batch_size, metrics, use_actor=True):
        dataloader = DataLoader(dataset=test_set, batch_size=batch_size)
        self.model.eval()
        r = {
            "_meta": {"type": "client_validation"},
            "E": round_number,
            "Length": 0,
            "Loss": 0,
        }
        for name in metrics:
            r[name] = 0
        
        with torch.no_grad():
            for _, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model.to(self.device)(data)
                r["Loss"] += self.loss_func(output, target).item() * len(target)
                r["Length"] += len(target)
                
                for name, metric in metrics.items():
                    r[name] += metric(output, target) * len(target)
        
        for name in metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]
        
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in metrics)
            + "\n"
        )
        return r
    
    def local_training(self, num_rounds: int, use_actor: bool, data_batches: list) -> None:
        r''' Local optimizaiton of the ``client``. Byzantine clients can overwrite this method to perform adversarial attack.
        
            :param num_rounds: Number of local optimization steps.
            :type num_rounds: int
            :param use_actor: Specifying the training mode, it should be ``True`` if you use ``Trainer Mode``
            :type use_actor: bool
            :param data_batches: A list of training batches for local training.
            :type data_batches: list
        '''
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
            self.optimizer.step()
        
        self.model = model
        update = (self._get_para(current=True) - self._get_para(current=False))
        self._save_update(update)
    
    def get_gradient(self) -> torch.Tensor:
        return self._get_saved_grad()
    
    def get_update(self) -> torch.Tensor:
        r''' Return the saved update of local optimization, represented as a vector.
        '''
        return torch.nan_to_num(self._get_saved_update())
    
    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end
    
    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state["saved_grad"] = torch.clone(p.grad).detach()
    
    def _save_update(self, update: torch.Tensor) -> None:
        self.state['saved_update'] = update.detach()
    
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


class _ClientWithMomentum(BladesClient):
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


class ByzantineClient(BladesClient):
    r"""Base class for Byzantine clients.
    
            .. note::
                Your Byzantine clients should also subclass this class, and overwrite ``local_training`` and
                ``omniscient_callback`` to customize your attack.
                
        """
    _is_byzantine = True
    def __int__(self, *args, **kwargs):
        super(ByzantineClient).__init__(*args, **kwargs)
    
    def omniscient_callback(self, simulator):
        r"""A method that will be registered by the simulator and execute after each communication round.
            It allows a Byzantine client has full knowledge of the training system, e.g., updates from all
            clients. Your Byzantine client can overwrite this method to access information from the server
            and other clients.
            
            :param simulator: The running simulator.
            :type simulator: Simulator
        """
        pass
