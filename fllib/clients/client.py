from typing import Iterable

import torch
from ray.rllib.utils import force_list
from ray.rllib.utils.from_config import from_config

from fllib.clients.callbacks import ClientCallbackList, ClientCallback
from fllib.constants import CLIENT_UPDATE
from fllib.core.execution.session import get_session

from .client_config import ClientConfig


class Client:
    def __init__(self, client_config: ClientConfig) -> None:
        self.config = client_config
        self.callbacks = None
        self._train_round = 0

        callback = from_config(self.config.callbacks_config)
        self.callbacks = ClientCallbackList(force_list(callback))
        self.callbacks.setup(self)

    def setup(self):
        sess = get_session(self.client_id)
        sess.task.init_optimizer(self.config.lr, self.config.momentum)

    @property
    def client_id(self):
        """Returns the ID of the client as specified in its configuration.

        Returns:
            The ID of the client.
        """
        return self.config.uid

    def set_callbacks(self, callbacks: ClientCallback):
        self.callbacks = ClientCallbackList(force_list(callbacks))

    def add_callback(self, callback: ClientCallback):
        self.callbacks.append(callback)

    def evaluate(self, test_loader):
        sess = get_session(self.client_id)
        return sess.task.evaluate(test_loader)

    def train_one_round(
        self,
        data_reader: Iterable,
    ):
        self._train_round += 1
        sess = get_session(self.client_id)
        sess.task.zero_psudo_grad()
        num_batches = self.config.num_batch_per_round
        running_loss = 0
        for _ in range(num_batches):
            data, target = data_reader.get_next_train_batch()
            data, target = self.callbacks.on_train_batch_begin(data, target)
            loss = sess.task.train_one_batch(
                data, target, self.callbacks.on_backward_end
            )
            running_loss += loss

        pseudo_grad = sess.task.compute_psudo_grad()
        self.pseudo_grad_vec = torch.cat([t.view(-1) for t in pseudo_grad.values()])

        avg_loss = running_loss / num_batches
        self.callbacks.on_train_round_end()

        round_result = {
            "id": self.client_id,
            CLIENT_UPDATE: self.pseudo_grad_vec,
            "avg_loss": avg_loss,
        }
        return round_result
