from typing import Optional

from ray.rllib.utils.from_config import from_config

from fllib.clients import Client


class ClientManager:
    def __init__(self, client_ids, client_config) -> None:
        self._clients = []
        self._client_map = {}

        for client_id in client_ids:
            config = client_config.client_id(client_id)
            self.add_client(config=config)

    @property
    def num_clients(self) -> int:
        """Returns the number of clients."""
        return len(self._clients)

    @property
    def clients(self) -> list:
        """Returns the list of clients."""
        return self._clients

    def add_client(
        self,
        client: Optional[Client] = None,
        config: Optional[dict] = None,
    ):
        if config is None and client is None:
            raise ValueError("client_cls and client cannot be None at the same time")
        if config is not None and client is not None:
            raise ValueError("client_cls and client cannot be both provided")
        if config is not None:
            client = from_config(config.cls, {"client_config": config})

        self._clients.append(client)
        self._client_map[client.client_id] = client
        return client

    def get_client_by_id(self, client_id: int) -> Client:
        """Returns the client with the given ID."""
        return self._client_map[client_id]
