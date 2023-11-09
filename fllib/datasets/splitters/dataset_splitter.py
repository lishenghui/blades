import itertools
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Iterator, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from fllib.datasets.clientdataset import ClientDataset


# TODO: Rename to DatasetPartitioner, which sounds more appropriate.
class DatasetSplitter(ABC):
    """An abstract base class for dataset splitting strategies that considers
    random states from both NumPy and PyTorch."""

    def __init__(
        self,
        num_clients: int,
        random_seed: int = 123,
        client_id_generator: Callable[[], Iterator] = None,
    ):
        """Initializes the dataset splitter with the number of clients and an
        optional random seed.

        Args:
            num_clients: The number of clients to split the data for.
            random_seed: An optional random seed for reproducibility.
            client_id_generator: An optional generator for creating client IDs.
        """
        self.num_clients = num_clients
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)  # Set NumPy random seed
            torch.manual_seed(random_seed)  # Set PyTorch random seed

        self.client_id_generator = (
            client_id_generator or self._default_client_id_generator()
        )

    def generate_subsets(self, dataset: Dataset) -> Dict[str, Subset]:
        """Generates subsets from a single dataset.

        Args:
            dataset: The dataset to be split.

        Returns:
            A dictionary with client IDs as keys and corresponding subsets as values.
        """
        subsets = self.split_dataset(
            dataset
        )  # Pass None for test_dataset if only one dataset is provided
        client_ids = self.generate_client_ids()
        return dict(zip(client_ids, subsets))

    def generate_paired_subsets(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> Dict[str, Tuple[Subset, Subset]]:
        """Generates paired subsets from two datasets that may interact with
        each other.

        Args:
            train_dataset: The training dataset to be split.
            test_dataset: The testing dataset to be split.

        Returns:
            A dictionary with client IDs as keys and tuples of corresponding training
            and testing subsets as values.
        """
        train_subsets, test_subsets = self.split_datasets(train_dataset, test_dataset)
        client_ids = self.generate_client_ids()
        return dict(zip(client_ids, zip(train_subsets, test_subsets)))

    def generate_client_datasets(
        self, train_dataset: Dataset, test_dataset: Dataset, **kwargs
    ):
        """Generates client datasets from two datasets that may interact with
        each other.

        Args:
            train_dataset: The training dataset to be split.
            test_dataset: The testing dataset to be split.

        Returns:
            A list of ClientDataset instances.
        """
        client_datasets = []
        paired_subsets = self.generate_paired_subsets(train_dataset, test_dataset)
        for client_id, (train_subset, test_subset) in paired_subsets.items():
            client_datasets.append(
                ClientDataset(
                    uid=client_id,
                    train_set=train_subset,
                    test_set=test_subset,
                    **kwargs,
                )
            )
        return client_datasets

    @abstractmethod
    def split_dataset(self, dataset: Dataset) -> List[Subset]:
        """Split a single dataset into multiple subsets, each keyed by a unique
        client_id.

        Args:
            dataset (Dataset): The dataset to be split.

        Returns:
            Dict[str, Subset]: A dictionary where the key is a string client_id and the
                               value is a Subset.
        """

    @abstractmethod
    def split_datasets(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> List[Tuple[Subset, Subset]]:
        """Split two datasets (e.g., training and testing datasets) into
        multiple pairs of subsets, each keyed by a unique client_id.

        Args:
            train_dataset (Dataset): The training dataset to be split.
            test_dataset (Dataset): The testing dataset to be split.

        Returns:
            Dict[str, Tuple[Subset, Subset]]: A dictionary where the key is a string
               client_id and the value is a tuple of two Subsets (training and testing).
        """

    @staticmethod
    def _default_client_id_generator():
        """A default generator for client IDs that yields sequential
        numbers."""
        return (f"client_{i}" for i in itertools.count(1))

    def generate_client_ids(self) -> List[Any]:
        """Generate a list of client IDs using the specified client ID
        generator."""
        return [next(self.client_id_generator) for _ in range(self.num_clients)]
