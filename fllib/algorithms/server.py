from typing import Dict, Optional, List

import torch
from ray.rllib.policy.torch_mixins import LearningRateSchedule
from ray.rllib.utils import force_list
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.typing import (
    ResultDict,
)
from torch import Tensor

from fllib.aggregators import Mean
from fllib.tasks import TaskSpec, Task
from fllib.types import ModelWeights


class _Base:
    def on_global_var_update(self, global_vars):
        pass


class Server(LearningRateSchedule, _Base):
    r"""Simulating the server of the federated learning system.

    :ivar aggregator: a callable which takes a list of tensors and returns
            an aggregated tensor.
    :vartype aggregator: callable

    :param  optimizer: The global optimizer, which can be any optimizer
    from Pytorch.
    :type optimizer: torch.optim.Optimizer
    :param model: The global global_model
    :type model: torch.nn.Module
    :param aggregator: a callable which takes a list of tensors and returns
            an aggregated tensor.
    :type aggregator: callable
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        device: str,
        optimizer_config: dict,
        aggregator_config: dict,
    ):
        lr = optimizer_config.get("lr", 0.1)
        lr_schedule = optimizer_config.get("lr_schedule", None)
        momentum = optimizer_config.get("momentum", 0)
        self.config = {}
        LearningRateSchedule.__init__(self, lr, lr_schedule)
        self.device = device
        self._task = task_spec.build(self.device)
        self._model = self._task.model
        self._optimizers = force_list(
            torch.optim.SGD(
                self._model.parameters(),
                lr=lr,
                momentum=momentum,
            )
        )
        self._aggregator = self._create_aggregator(aggregator_config)
        self._states: Dict[str, Optional[Tensor]] = {}

    def _create_aggregator(self, config: dict):
        aggregator = from_config(Mean, config)
        return aggregator

    @property
    def aggregator(self):
        return self._aggregator

    @property
    def task(self) -> Task:
        """Task instance."""
        return self._task

    def register_state(self, name: str, tensor: Optional[Tensor]) -> None:
        self._states[name] = tensor

    def get_state(self, name: str) -> ModelWeights:
        return self._states[name]

    def set_state(self, name: str, tensor: Optional[Tensor]) -> None:
        self._states[name] = tensor

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.
        It should be called before assigning pseudo-gradient.

        Args:
            set_to_none: See `Pytorch documentation <https://pytorch.org/docs/s
            table/generated/torch.optim.Optimizer.zero_grad.html>`_.
        """
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def get_global_model(self):
        return self._model

    def step(
        self,
        local_updates: List[Tensor],
        global_vars: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ResultDict:
        self.update_global_model(local_updates)
        super().on_global_var_update(global_vars)
        return {"server lr": [opt.param_groups[0]["lr"] for opt in self._optimizers]}

    def update_global_model(self, local_updates: List[Tensor]) -> None:
        r"""Apply a step of global optimization.

            .. note::
                The input should be a ``Tensor``, which will be converted to
                ``pseudo-gradient`` layer by layer.

        Args:
            update: The aggregated update.
        """
        update = self._aggregator(local_updates)
        self.zero_grad()
        beg = 0
        for param in self._model.parameters():
            if not param.requires_grad:
                continue
            end = beg + len(param.data.view(-1))
            psudo_grad = update[beg:end].reshape_as(param.data)
            param.grad = -psudo_grad.clone().detach().to(param.device)
            beg = end
        for opt in self._optimizers:
            opt.step()
