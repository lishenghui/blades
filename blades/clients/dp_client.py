import torch

from blades.clients import Client
from fllib.clients.callbacks import ClientCallback
from fllib.clients.client_config import ClientConfig
from fllib.types import NotProvided
from fllib.utils.torch_utils import clip_tensor_norm_


class DPCliengConfig(ClientConfig):
    def __init__(self, client_class=None) -> None:
        super().__init__(class_specifier=client_class or DPClient)
        self.clip_threshold = 100.0
        self.noise_factor = 0

    def training(
        self,
        num_batch_per_round: int = NotProvided,
        lr: float = NotProvided,
        clip_threshold=NotProvided,
        noise_factor=NotProvided,
        momentum: float = NotProvided,
    ) -> ClientConfig:
        super().training(num_batch_per_round, lr, momentum=momentum)
        if clip_threshold is not NotProvided:
            self.clip_threshold = clip_threshold
        if noise_factor is not NotProvided:
            self.noise_factor = noise_factor
        return self


class _DPCallback(ClientCallback):
    def __init__(self, clip_threshold, noise_factor) -> None:
        self._clip_threshold = clip_threshold
        self._noise_factor = noise_factor

    def on_train_round_end(self):
        update = self._client.pseudo_grad_vec
        clip_tensor_norm_(update, max_norm=self._clip_threshold)

        sigma = self._noise_factor
        noise = torch.normal(mean=0.0, std=sigma, size=update.shape).to(update.device)
        update += noise


class DPClient(Client):
    def __init__(self, client_config: ClientConfig, is_malicious=False):
        super().__init__(client_config, is_malicious)
        self.add_callback(
            _DPCallback(client_config.clip_threshold, client_config.noise_factor)
        )
