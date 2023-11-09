import copy
import logging
import threading
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Union, Callable

import torch
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    OverrideToImplementCustomLogic,
    is_overridden,
)
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    ModelWeights,
    TensorType,
)

from fllib.models.catalog import ModelCatalog

if TYPE_CHECKING:
    from ray.rllib.evaluation import Episode  # noqa

logger = logging.getLogger(__name__)


class TaskSpec:
    def __init__(self, task_class=None, alg_config=None):
        # If None, use the Algorithm's default policy class stored under
        self.task_class = task_class

        self.config = alg_config

    def __eq__(self, other: "TaskSpec"):
        return self.task_class == other.task_class and self.config == other.config

    def build(self, device: str) -> "Task":
        """Builds a Task instance from this TaskSpec.

        Args:
            config (dict): The main Algorithm config dict.

        Returns:
            Task: The Task instance.
        """
        return from_config(self.task_class, device=device, alg_config=self.config)


@DeveloperAPI
class Task:
    """PyTorch specific Policy class to use with RLlib."""

    @DeveloperAPI
    def __init__(
        self,
        device: str,
        alg_config: AlgorithmConfigDict,
    ):
        """Initializes a Task instance.

        Args:
            device: The device.
            alg_config: The Algorithm's config dict.
        """
        self._loss_initialized = False
        self.config = alg_config
        self.device = device
        self._model = self._init_model().to(self.device)
        self._lock = threading.RLock()
        self._optimizers = None
        self._saved_state_dict = None

        self._metrics = None

    @property
    def model(self):
        return self._model

    def loss_initialized(self):
        return self._loss_initialized

    @staticmethod
    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def loss(
        model: torch.nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss function.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            Loss tensor given the input batch.
        """
        raise NotImplementedError

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def make_model(self) -> torch.nn.Module:
        """Create model.

        Note: only one of make_model or make_model_and_action_dist
        can be overridden.

        Returns:
            Torch model.
        """
        return None

    def zero_psudo_grad(self):
        self._saved_state_dict = copy.deepcopy(self._model.state_dict())

    def compute_psudo_grad(self):
        pseudo_grad = {}
        for name, param in self._model.named_parameters():
            pseudo_grad[name] = param.data - self._saved_state_dict[name]

        self._model.load_state_dict(self._saved_state_dict)
        return pseudo_grad

    def train_one_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        on_backward_end: Callable = None,
    ):
        self._model.train()
        for _, opt in enumerate(self._optimizers):
            opt.zero_grad()
        loss = self.loss(self._model, data, target)

        loss.backward()
        if on_backward_end:
            on_backward_end(self)
        for _, opt in enumerate(self._optimizers):
            opt.step()
        return loss.item()

    @abstractmethod
    def init_metrics(self):
        raise NotImplementedError

    def evaluate(self, test_loader):
        if self._metrics is None:
            self._metrics = self.init_metrics()
        device = next(self._model.parameters()).device
        self._metrics.to(device)
        self._metrics.reset()
        self._model.eval()

        result = {"length": 0}
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self._model(data)
                result["length"] += len(target)
                self._metrics(output, target)
        result.update(self._metrics.compute())
        return result

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def init_optimizer(
        self,
        lr=1.0,
        momentum=0.0,
    ) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:
        """Custom the local PyTorch optimizer(s) to use.

        Returns:
            The local PyTorch optimizer(s) to use for this Policy.
        """
        optimizers = [
            torch.optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        ]
        self._optimizers = force_list(optimizers)
        return optimizers

    def get_optimizer_states(self):
        return [opt.state_dict() for opt in self._optimizers]

    def set_optimizer_states(self, states):
        for opt, state in zip(self._optimizers, states):
            opt.load_state_dict(state)

    def _init_model(self):
        if is_overridden(self.make_model):
            model = self.make_model()
        else:
            model = ModelCatalog.get_model(self.config["global_model"])

        return model

    @DeveloperAPI
    def get_weights(self) -> ModelWeights:
        return {
            k: v.cpu().detach().numpy() for k, v in self._model.state_dict().items()
        }

    @DeveloperAPI
    def set_weights(self, weights: ModelWeights) -> None:
        device = next(self._model.parameters()).device
        weights = convert_to_torch_tensor(weights, device=device)
        self._model.load_state_dict(weights)

    @DeveloperAPI
    def import_model_from_h5(self, import_file: str) -> None:
        """Imports weights into torch model."""
        return self._model.import_from_h5(import_file)
