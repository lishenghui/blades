import random
from typing import List

from torch.utils.data import ConcatDataset

from fllib.datasets import clientdataset


class FLDataset:
    def __init__(self, client_datasets) -> None:
        self.client_datasets = client_datasets
        # Create a dictionary to map client_id to ClientDataset for O(1) lookup
        self.client_datasets_dict = {
            client_dataset.uid: client_dataset for client_dataset in client_datasets
        }

    def __len__(self):
        return len(self.client_datasets)

    @property
    def client_ids(self) -> List[str]:
        """Returns the list of client IDs in the FLDataset.

        Returns:
            List[str]: The list of client IDs in the FLDataset.
        """
        return list(self.client_datasets_dict.keys())

    @property
    def train_client_ids(self) -> List[str]:
        """Returns the list of client IDs in the FLDataset that have training
        data.

        Returns:
            List[str]: The list of client IDs in the FLDataset that have training data.
        """
        return [
            client_id
            for client_id, client_dataset in self.client_datasets_dict.items()
            if client_dataset.train_set_size > 0
        ]

    @property
    def test_client_ids(self) -> List[str]:
        return [
            client_id
            for client_id, client_dataset in self.client_datasets_dict.items()
            if client_dataset.test_set_size > 0
        ]

    def split(self, num_shards: int) -> List["FLDataset"]:
        """Splits the FLDataset into a specified number of shards, each
        containing a subset of clients.

        :param num_shards: Number of shards to create.
        :return: A list of FLDataset instances, each with a subset of the clients.
        """
        # Shuffle client datasets to ensure randomness
        random.shuffle(self.client_datasets)

        # Split the client datasets into num_shards shards
        shard_size = len(self.client_datasets) // num_shards
        remaining = len(self.client_datasets) % num_shards
        shards = []

        start = 0
        for _ in range(num_shards):
            end = start + shard_size + (1 if remaining > 0 else 0)
            shards.append(self.client_datasets[start:end])
            start = end
            remaining -= 1

        # Create a new FLDataset instance for each shard
        return [FLDataset(shard) for shard in shards]

    def to_torch_datasets(self) -> ConcatDataset:
        """Converts the FLDataset into torch ConcatDatasets.

        :return: torch ConcatDatasets.
        """
        trainset = ConcatDataset(
            [client_set.train_set for client_set in self.client_datasets]
        )
        testset = ConcatDataset(
            [client_set.test_set for client_set in self.client_datasets]
        )
        return trainset, testset

    def get_client_dataset(self, client_id: str) -> "clientdataset.ClientDataset":
        """Returns the ClientDataset instance corresponding to the specified
        client ID with O(1) complexity.

        :param client_id: The client ID.
        :return: The ClientDataset instance corresponding to the specified client ID.
        """
        try:
            return self.client_datasets_dict[client_id]
        except KeyError as e:
            raise ValueError(f"Client ID {client_id} not found.") from e

    def __repr__(self) -> str:
        return (
            f"Train client ids: {self.train_client_ids.__repr__()} \n"
            f"Test client ids: {self.test_client_ids.__repr__()}"
        )
