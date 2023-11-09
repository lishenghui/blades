from typing import Callable, Iterator, List

import numpy as np
from torch.utils.data import Dataset, Subset

from .dataset_splitter import DatasetSplitter


class ShardSplitter(DatasetSplitter):
    def __init__(
        self,
        num_clients: int,
        random_seed: int = 123,
        client_id_generator: Callable[[], Iterator] = None,
        num_shards: int = 4,
    ):
        super().__init__(num_clients, random_seed, client_id_generator)

        assert num_shards >= num_clients, (
            "Number of shards cannot be smaller than " "clients"
        )
        self.num_shards = num_shards

    def split_dataset(self, dataset) -> List[Subset]:
        # Sort data by label
        targets = dataset.targets
        indices = np.argsort(targets)

        # Calculate shard sizes
        num_indices = len(indices)
        base_shard_size = num_indices // self.num_shards
        extra = num_indices % self.num_shards

        # Initialize shards
        shards_indices = []

        # Assign indices to shards with extra indices distributed among the first
        # few shards
        start_idx = 0
        for i in range(self.num_shards):
            end_idx = start_idx + base_shard_size + (1 if i < extra else 0)
            shards_indices.append(indices[start_idx:end_idx])
            start_idx = end_idx

        # Now we can shuffle the shards to ensure random distribution if required
        np.random.shuffle(shards_indices)

        # Assign shards to clients evenly
        client_data_indices = [[] for _ in range(self.num_clients)]
        for shard_indices in shards_indices:
            client_idx = np.argmin([len(indices) for indices in client_data_indices])
            client_data_indices[client_idx].extend(shard_indices)

        # Create client datasets
        client_datasets = [
            Subset(dataset, subset_indices) for subset_indices in client_data_indices
        ]

        # Check the number of clients
        assert (
            len(client_datasets) == self.num_clients
        ), "Number of clients is not equal to expected."

        return client_datasets

    def split_datasets(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> tuple[list[Subset], list[Subset]]:
        # Use the split_dataset method to split both the training and testing datasets
        train_subsets = self.split_dataset(train_dataset)
        test_subsets = self.split_dataset(test_dataset)

        return train_subsets, test_subsets
