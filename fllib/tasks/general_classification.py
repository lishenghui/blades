from functools import partial

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MetricCollection, Metric

from fllib.tasks.task import Task


class CustomCrossEntropy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            distributed_available_fn=lambda: False,
        )
        self.add_state("ce_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_examples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        loss = self.ce_loss_func(y_pred, y_true)
        self.ce_loss += loss
        self.num_examples += y_true.numel()

    def compute(self):
        ce_loss = self.ce_loss / self.num_examples
        return ce_loss.item()


class GeneralClassification(Task):
    _num_classes = None

    @staticmethod
    def loss(
        model: torch.nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
    ):
        # Get the device on which the model is located
        device = next(model.parameters()).device
        data, target = data.to(device), target.to(device)
        output = model(data)

        if GeneralClassification._num_classes is None:
            GeneralClassification._num_classes = output.shape[1]
        loss = F.cross_entropy(output, target)
        return torch.clamp(loss, 0, 1e6)

    @property
    def num_classes(self):
        if GeneralClassification._num_classes is None:
            raise ValueError(
                "The number of classes is not defined yet, you may need to run a"
                " forward pass first, so that the number of outputs is known."
            )
        return GeneralClassification._num_classes

    def init_metrics(self):
        # Define a base partial function with shared arguments
        metrics = MetricCollection(
            {
                "ce_loss": CustomCrossEntropy(),
            }
        )
        if self._num_classes:
            base_accuracy = partial(
                torchmetrics.Accuracy,
                task="multiclass",
                num_classes=self._num_classes,
                distributed_available_fn=lambda: False,
            )
            for k in range(1, self._num_classes, 2):
                metrics[f"acc_top_{k}"] = partial(base_accuracy, top_k=k)()
        return metrics
