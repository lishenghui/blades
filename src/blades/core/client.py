import logging
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from blades.utils.torch_utils import clip_tensor_norm_


class BladesClient(object):
    _is_byzantine: bool = False
    r"""Base class for all clients.

        .. note::
            Your honest clients should also subclass this class.
    Args:
        id (str): a unique id of the client.
        device (str): target device if specified, all parameters will be
                        copied to that device.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        self._state = defaultdict(dict)
        self.device = device
        self._is_trusted: bool = False

        self._json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")

        self.set_id(id)

    def set_id(self, id: str) -> None:
        r"""Sets the unique id of the client."""

        # if not isinstance(id,str):
        #     raise TypeError(f'Client _id must be str, but got {type(id)}')
        self._id = id

    def id(self) -> str:
        r"""Returns the unique id of the client.

        :Example:

        >>> from blades.core.client import BladesClient
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

    def set_model_ref(self, model):
        """Copy an existing model reference.

        Args:
            model: ``Torch`` model

        Returns:
        """
        self.global_model = model

    def set_loss(self, loss_func="crossentropy"):
        if loss_func == "crossentropy":
            self.loss_func = nn.modules.loss.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return "BladesClient"

    def on_train_round_begin(self, globel_model=None) -> None:
        """Called at the beginning of each local training round in
        `local_training` methods.

        Subclasses should override for any actions to run.

        :param logs: Dict. Aggregated metric results up until this batch.
        """
        self._save_para(globel_model)

    def on_train_round_end(
        self, dp=False, clip_threshold=None, noise_factor=None
    ) -> None:
        """Called at the end of local optimization."""
        update = self._get_para(current=True) - self._get_para(current=False)
        if dp:
            assert clip_threshold is not None
            clip_tensor_norm_(update, max_norm=clip_threshold)

            sigma = noise_factor
            noise = torch.normal(
                mean=0.0,
                std=sigma,
                size=update.shape,
            ).to(update.device)
            update += noise
        self.save_update(update)
        self.global_model = None
        self._state["saved_para"].clear()

    def on_train_batch_begin(self, data, target, logs=None):
        """Called at the beginning of a training batch in `local_training`
        methods.

        Subclasses should override for any actions to run.

        Args:
            data: input of the batch data.
            target: target of the batch data.
            logs: Dict. Aggregated metric results up until this batch.
        """
        return data, target

    def local_training(self, data_batches: list, opt) -> None:
        r"""Local optimizaiton of the ``client``. Byzantine input can override
        this method to perform adversarial attack.

        Args:
            data_batches: A list of training batches for local training.
            opt: Optimizer.
        """
        for data, target in data_batches:
            data, target = data.to(self.device), target.to(self.device)
            data, target = self.on_train_batch_begin(data=data, target=target)
            opt.zero_grad()

            output = self.global_model(data)
            # Clamp loss value to avoid possible 'Nan' gradient with some
            # attack types.
            loss = torch.clamp(self.loss_func(output, target), 0, 1e6)
            loss.backward()
            opt.step()

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
            for _, (data, target) in enumerate(dataloader):
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
        return torch.nan_to_num(self._get_saved_update())

    def save_update(self, update: torch.Tensor) -> None:
        r"""Sets the update of the client,.

        Args:
        update: a vector of local update
        """
        self._state["saved_update"] = torch.clone(update).detach()

    def _get_saved_update(self) -> torch.Tensor:
        return torch.Tensor(self._state["saved_update"])

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
    override ``local_training`` and ``omniscient_callback`` to     customize
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
