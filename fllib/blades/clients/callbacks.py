import torch

from fllib.clients.callbacks import ClientCallback


class ClippingCallback(ClientCallback):
    def __init__(self, clip_threshold) -> None:
        self._clip_threshold = clip_threshold

    def on_backward_end(self, task):
        super().on_backward_end(task)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            task._model.parameters(), self._clip_threshold
        )
        if gradient_norm > self._clip_threshold:
            print(
                f"Round: {self._client._train_round}, Client: {self._client.client_id}"
                f"梯度 {gradient_norm} 范数超过阈值，可能存在梯度爆炸问题"
            )
