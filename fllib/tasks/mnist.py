import torch
from fllib.tasks.task import Task
import torch.nn.functional as F


class MNIST(Task):
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
        loss = F.cross_entropy(output, target)
        return torch.clamp(loss, 0, 1e6)
