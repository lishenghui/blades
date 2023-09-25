import copy
from typing import Iterable, Type, Optional

import fllib.clients
from fllib.clients import ClientConfig


class Client(fllib.clients.Client):
    """Represents a client.

    Attributes:
        _is_malicious (bool): Whether the client is malicious or not.
    """

    def __init__(
        self,
        client_config: ClientConfig,
        is_malicious: bool = False,
    ):
        """Initializes a new client."""
        super().__init__(client_config=client_config)
        self._is_malicious = is_malicious
        self._is_under_attack = False

    @property
    def is_malicious(self):
        """Returns whether the client is malicious or not."""
        return self._is_malicious

    def attack(self):
        """Attacks the client."""
        if not self.is_malicious:
            raise ValueError(
                "The client is not inherently malicious, but it needs to be transformed"
                "using the `to_malicious` method prior to launching an attack."
            )
        self._is_malicious = True

    def restore(self):
        """Restores the client."""
        self._is_under_attack = False

    def to_malicious(
        self,
        target_cls: Optional[Type] = None,
        callbacks_cls: Optional[Type] = None,
        local_training: Optional[bool] = True,
    ):
        """Turns the client to a malicious one."""

        client_copy = copy.deepcopy(self)
        new_object = ClientProxy(self, target_cls, local_training=local_training)

        self.__class__ = new_object.__class__
        self.__init__(client_copy, target_cls, local_training=local_training)
        self.attack()
        if callbacks_cls:
            self.set_callbacks(callbacks_cls())


class ClientProxy:
    """Monkey patches a client to make it malicious.

    This is used to simulate a malicious client. The client will act maliciously after
    the `attack` method is called, while it will act benignly after the `restore`
    method. To simulate such a behavior.
    """

    is_malicious = True

    def __init__(
        self,
        client: fllib.clients.Client,
        malicious_cls: Type = None,
        local_training: Optional[bool] = False,
    ) -> None:
        self._is_under_attack = False
        self._local_training = local_training
        self.benign_client = client

        self.malicious_client = (
            malicious_cls(client_config=client.config) if malicious_cls else client
        )

    @property
    def client_id(self):
        """Returns the client ID."""
        return self.benign_client.client_id

    def attack(self):
        """Attacks the client."""
        self._is_under_attack = True

    def restore(self):
        """Restores the client."""
        self._is_under_attack = False

    def set_callbacks(self, callbacks):
        if self._is_under_attack:
            self.malicious_client.add_callback(callbacks)
        else:
            self.benign_client.add_callback(callbacks)

    def train_one_round(self, data_reader: Iterable):
        """Trains the client for one round."""
        if not self._local_training:
            return {
                "id": self.client_id,
            }
        if self._is_under_attack:
            return self.malicious_client.train_one_round(data_reader=data_reader)
        return self.benign_client.train_one_round(data_reader=data_reader)

    def evaluate(self, test_loader):
        if self._is_under_attack:
            return self.malicious_client.evaluate(test_loader)
        else:
            return self.benign_client.evaluate(test_loader)
