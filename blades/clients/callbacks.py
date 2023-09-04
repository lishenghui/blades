import torch

from fllib.clients.callbacks import ClientCallback


class ClippingCallback(ClientCallback):
    def __init__(self, clip_threshold) -> None:
        self._clip_threshold = clip_threshold

    def on_backward_end(self, task):
        super().on_backward_end(task)

        _ = torch.nn.utils.clip_grad_norm_(
            task._model.parameters(), self._clip_threshold
        )
