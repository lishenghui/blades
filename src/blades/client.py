import copy
import logging
from collections import defaultdict
from typing import Union, Tuple, Optional

import ray.train as train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BladesClient(object):
    r"""Base class for all input.
    
        .. note::
            Your honest input should also subclass this class.
    
        :param id: a unique id of the client.
        :param device:  if specified, all parameters will be copied to that device
    """
    
    _is_byzantine: bool = False
    _is_trusted: bool = False
    device: str = 'cpu'
    _state = defaultdict(dict)
    def __init__(
            self,
            id: Optional[str] = None,
            device: Optional[Union[torch.device, str]] = 'cpu',
    ):
        self.set_id(id)
        self.device = device
        
        self._running = {}
        
        self._json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
    
    def set_id(self, id: str) -> None:
        r"""Sets the unique id of the client.
        """
        # if not isinstance(id,str):
        #     raise TypeError(f'Client _id must be str, but got {type(id)}')
        self._id = id
    
    def id(self) -> str:
        r"""Returns the unique id of the client.
        
        :Example:
        
        >>> from blades.client import BladesClient
        >>> client = BladesClient(id='1')
        >>> client.id()
        '1'

        """
        return self._id
    
    def getattr(self, attr):
        return getattr(self, attr)
    
    def is_byzantine(self):
        r"""Return a boolean value specifying if the client is Byzantine"""
        return self._is_byzantine
    
    def is_trusted(self):
        return self._is_trusted
    
    def trust(self, trusted: Optional[bool]=True) -> None:
        r"""
        Trusts the client as an honest participant. This property is useful
        for trust-based algorithms.
        """
        self._is_trusted = trusted
    
    def set_model(self, model: nn.Module, opt: type(torch.optim.Optimizer), lr: float) -> None:
        r''' Deep copy the given model to the client.
        
            .. note::
                To improve the scalability, this API may be removed in the future,
                
            :param model: a Torch model for current client.
            :param opt: client optimizer
            :param lr:  local learning rate
        '''
        self.model = copy.deepcopy(model)
        self.optimizer = opt(self.model.parameters(), lr=lr)
    
    def set_loss(self, loss_func='crossentropy'):
        if loss_func == 'crossentropy':
            self.loss_func = nn.modules.loss.CrossEntropyLoss()
        else:
            raise NotImplementedError
    
    def set_para(self, model):
        self.model.load_state_dict(model.state_dict())
    
    def __str__(self) -> str:
        return "BladesClient"
    
    def on_train_round_start(self) -> None:
        """Called at the beginning of each local training round in `local_training` methods.

        Subclasses should override for any actions to run.

        :param logs: Dict. Aggregated metric results up until this batch.
        """
        self.model = self.model.to(self.device)
        self.model.train()

    def on_train_batch_begin(self, data, target, logs=None):
        """Called at the beginning of a training batch in `local_training` methods.

         Subclasses should override for any actions to run.

         :param data: input of the batch data.
         :param target: target of the batch data.
         :param logs: Dict. Aggregated metric results up until this batch.
         """
        return data, target
        
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
        
        self._json_logger.info(r)
        self.debug_logger.info(
            f"\n=> Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in metrics)
            + "\n"
        )
        return r
    
    def local_training(self, num_rounds: int, use_actor: bool, data_batches: list) -> None:
        r''' Local optimizaiton of the ``client``. Byzantine input can overwrite this method to perform adversarial attack.
        
            :param num_rounds: Number of local optimization steps.
            :param use_actor: Specifyinmodelg the training mode, it should be ``True`` if you use ``Trainer Mode``
            :param data_batches: A list of training batches for local training.
        '''
        self._save_para()
        if use_actor:
            model = self.model
        else:
            model = train.torch.prepare_model(self.model)
        
        for data, target in data_batches:
            data, target = data.to(self.device), target.to(self.device)
            data, target = self.on_train_batch_begin(data=data, target=target)
            self.optimizer.zero_grad()
            
            output = model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
        
        self.model = model
        update = (self._get_para(current=True) - self._get_para(current=False))
        self.save_update(update)
    
    def get_update(self) -> torch.Tensor:
        r'''Returns the saved update of local optimization, represented as a vector.
        '''
        return torch.nan_to_num(self._get_saved_update())
    
    def save_update(self, update: torch.Tensor) -> None:
        r"""Sets the update of the client,.
        
        :param update: a vector of local update
        """
        self._state['saved_update'] = update.detach()

    def _get_saved_update(self) -> torch.Tensor:
        return self._state['saved_update']

    def _save_para(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                param_state = self._state[p]
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
                    param_state = self._state[p]
                    layer_parameters.append(param_state["saved_para"].data.view(-1))
        return torch.cat(layer_parameters).to('cpu')


class ByzantineClient(BladesClient):
    r"""Base class for Byzantine input.
    
            .. note::
                Your Byzantine input should also subclass this class, and overwrite ``local_training`` and
                ``omniscient_callback`` to customize your attack.
                
        """
    _is_byzantine = True
    
    def __int__(self, *args, **kwargs):
        super(ByzantineClient).__init__(*args, **kwargs)
    
    def omniscient_callback(self, simulator):
        r"""A method that will be registered by the simulator and execute after each communication round.
            It allows a Byzantine client has full knowledge of the training system, e.g., updates from all
            input. Your Byzantine client can overwrite this method to access information from the server
            and other input.
            
            :param simulator: The _running simulator.
            :type simulator: Simulator
        """
        pass
