import logging
from collections import defaultdict
from typing import Optional, Generator, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BladesClient(object):
    r"""Base class for all clients.

    .. note::     Your honest clients should also subclass this class.
    """

    _is_byzantine: bool = False

    def __init__(
        self,
        id: Optional[str] = None,
        momentum: Optional[float] = 0.0,
        dampening: Optional[float] = 0.0,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        """
        Args:
            id (str): a unique id of the client.
            momentum (float, optional): momentum factor (default: 0)
            device (str): target device if specified, all parameters will be
                        copied to that device.
        """

        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        self.momentum = momentum
        self.momentum_buff = None
        self.dampening = dampening
        self._state = defaultdict(dict)
        self._is_trusted: bool = False
        self._json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        self.device = device
        self.set_id(id)

    def set_id(self, id: str) -> None:
        """Sets the unique id of the client.

        Args:
            id (str):a unique id of the client.
        """
        r""""""

        if not isinstance(id, (str, type(None))):
            raise TypeError(f"Client _id must be str or None, but got {type(id)}")
        self._id = id

    def id(self) -> str:
        r"""Returns the unique id of the client.

        :Example:

        >>> from blades.clients import BladesClient
        >>> client = BladesClient(id='1')
        >>> client.id()
        '1'
        """
        return self._id

    def getattr(self, attr):
        return getattr(self, attr)

    def is_byzantine(self):
        r"""Return a boolean value specifying if the client is Byzantine."""
        return self._is_byzantine

    def is_trusted(self):
        return self._is_trusted

    def trust(self, trusted: Optional[bool] = True) -> None:
        r"""Trusts the client as an honest participant. This property is useful
        for trust-based algorithms.

        Args:
            trusted: Boolean; whether the client is trusted; default to True.
        """
        self._is_trusted = trusted

    def set_global_model_ref(self, model):
        """Copy an existing global_model reference.

        Args:
            model: ``Torch`` global_model

        Returns:
        """
        self.global_model = model

    def set_loss(self, loss_func="cross_entropy"):
        if loss_func == "cross_entropy":
            self.loss_func = nn.modules.loss.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return "BladesClient"

    def on_train_round_begin(self) -> None:
        """Called at the beginning of each local training round in
        `train_global_model` methods.

        Subclasses should override for any actions to run.

        Returns:
        """

    def on_train_batch_begin(self, data, target):
        """Called at the beginning of a training batch in `train_global_model`
        methods.

        Subclasses should override for any actions to run.

        Args:
            data: input of the batch data.
            target: target of the batch data.
        """
        return data, target

    def on_backward_end(self):
        """A callback method called after backward and before parameter update.

        It is typically used to modify gradients.
        """
        pass

    def on_train_round_end(self):
        """A callback method called after local training.

        It is typically used to modify updates (i.e,. pseudo-gradient).
        """
        pass

    def train_global_model(
        self, train_set: Generator, num_batches: int, opt: torch.optim.Optimizer
    ) -> None:
        r"""Local optimization of the ``client``. Byzantine input can override
        this method to perform adversarial attack.

        Args:
            train_set: A list of training batches for local training.
            num_batches: Number of batches of local update.
            opt: Optimizer.
        """
        self._save_para(self.global_model)
        self.global_model.train()
        for i in range(num_batches):
            data, target = next(train_set)
            data, target = data.to(self.device), target.to(self.device)
            data, target = self.on_train_batch_begin(data=data, target=target)
            opt.zero_grad()

            output = self.global_model(data)
            # Clamp loss value to avoid possible 'Nan' gradient with some
            # attack types.
            loss = torch.clamp(self.loss_func(output, target), 0, 1e6)
            loss.backward()

            self.on_backward_end()
            opt.step()
        update = self._get_para(current=True) - self._get_para(current=False)

        self.update_buffer = torch.clone(update).detach()
        if self.momentum > 0.0:
            if self.momentum_buff is None:
                self.momentum_buff = torch.zeros_like(
                    self.update_buffer, device=self.update_buffer.device
                )
            self.momentum_buff.mul_(self.momentum).add_(
                self.update_buffer, alpha=1 - self.dampening
            )
            self.update_buffer = self.momentum_buff

        self.global_model = None
        self._state["saved_para"].clear()
        self.on_train_round_end()

    def train_personal_model(
        self, train_set: Generator, num_batches: int, global_state: Dict
    ) -> None:
        r"""Local optimization of the ``client``. Byzantine input can override
        this method to perform adversarial attack.

        Args:
            train_set: A generator of training set for local training.
        """
        pass

    def evaluate(self, round_number, test_set, batch_size, metrics):
        """Model evaluation.

        Args:
            round_number: Current global round.
            test_set: Data set for test.
            batch_size: Test batch size.
            metrics: Metrics.

        Returns:
        """

        dataloader = DataLoader(dataset=test_set, batch_size=batch_size)
        self.global_model.eval()
        r = {
            "_meta": {"type": "client_validation"},
            "E": round_number,
            "Length": 0,
            "Loss": 0,
        }
        for name in metrics:
            r[name] = 0

        with torch.no_grad():
            for (data, target) in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model.to(self.device)(data)
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

    def get_update(self) -> torch.Tensor:
        """Returns the saved update of local optimization, represented as a
        vector.

        Returns: a vector tensor of update parameters.
        """
        return torch.nan_to_num(self.update_buffer)

    def save_update(self, update: torch.Tensor) -> None:
        r"""Sets the update of the client.

        Args:
        update: a vector of local update
        """
        self.update_buffer = torch.clone(update).detach()

    def _save_para(self, model) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self._state["saved_para"][name] = (
                torch.clone(param.data).detach().to(self.device)
            )

    def _get_para(self, current=True) -> torch.Tensor:
        layer_parameters = []

        for name, param in self.global_model.named_parameters():
            if not param.requires_grad:
                continue
            if current:
                layer_parameters.append(param.data.view(-1))
            else:
                saved_param = self._state["saved_para"][name]
                layer_parameters.append(saved_param.data.view(-1))

        return torch.cat(layer_parameters)


class ByzantineClient(BladesClient):
    r"""Base class for Byzantine input.

    .. note::     Your Byzantine input should also subclass this class, and
    override ``train_global_model`` and ``omniscient_callback`` to customize
    your attack.
    """
    _is_byzantine = True

    def __int__(self, *args, **kwargs):
        super(ByzantineClient).__init__(*args, **kwargs)

    def omniscient_callback(self, simulator):
        r"""A method that will be registered by the simulator and execute after
        each communication round. It allows a Byzantine client has full
        knowledge of the training system, e.g., updates from all input. Your
        Byzantine client can override this method to access information from
        the server and other input.

        Args:
            simulator: The running simulator.
        """
        pass
