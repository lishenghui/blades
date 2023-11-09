from typing import List, Callable, Iterator, Any

import torch
from torch.utils.data import Dataset, Subset

from .dataset_splitter import DatasetSplitter


class DirichletSplitter(DatasetSplitter):
    def __init__(
        self,
        num_clients: int = 4,
        random_seed: int = 123,
        client_id_generator: Callable[[], Iterator] = None,
        alpha: float = 1.0,
        same_proportions: bool = True,
    ):
        super().__init__(num_clients, random_seed, client_id_generator)
        self.alpha = alpha
        self.same_proportions = same_proportions

    def split_dataset(self, dataset: Dataset) -> List[Subset]:
        subsets, _ = self._split_single_dataset(dataset)
        return subsets

    def split_datasets(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> tuple[list[Subset[Any]], list[Subset[Any]]]:
        # Split train dataset and get the class proportions
        train_subsets, class_proportions = self._split_single_dataset(train_dataset)

        # Optionally, use the same class proportions for test dataset
        if self.same_proportions:
            test_subsets, _ = self._split_single_dataset(
                test_dataset, class_proportions
            )
        else:
            test_subsets, _ = self._split_single_dataset(test_dataset)

        return train_subsets, test_subsets

    def _split_single_dataset(self, dataset: Dataset, class_proportions=None):
        targets = torch.tensor(dataset.targets)
        self._validate_dataset(targets)
        num_classes = len(torch.unique(targets))
        class_indices = self._get_class_indices(targets, num_classes)

        # Sample from Dirichlet distribution if class_proportions are not provided
        if class_proportions is None:
            class_proportions = torch.distributions.Dirichlet(
                torch.ones(self.num_clients) * self.alpha
            ).sample((num_classes,))

        client_indices = self._allocate_samples_to_clients(
            class_indices, class_proportions
        )
        return [
            Subset(dataset, indices) for indices in client_indices
        ], class_proportions

    def _validate_dataset(self, targets: torch.Tensor):
        assert (
            len(targets) >= self.num_clients
        ), "Not enough samples to distribute to each client"

    @staticmethod
    def _get_class_indices(
        targets: torch.Tensor, num_classes: int
    ) -> List[torch.Tensor]:
        return [torch.where(targets == i)[0] for i in range(num_classes)]

    def _allocate_samples_to_clients(
        self, class_indices: List[torch.Tensor], class_proportions: torch.Tensor
    ) -> List[List[int]]:
        # Sample from Dirichlet distribution for class proportions

        client_indices = [[] for _ in range(self.num_clients)]

        # Allocate samples to clients class by class
        for c, indices in enumerate(class_indices):
            shuffled_indices = indices[torch.randperm(len(indices))]
            allocations = self._calculate_allocations(
                shuffled_indices, class_proportions[c]
            )

            # Assign indices to clients based on allocations
            prev_proportion = 0
            for client_idx, allocation in enumerate(allocations):
                client_indices[client_idx].extend(
                    shuffled_indices[
                        prev_proportion : prev_proportion + allocation
                    ].tolist()
                )
                prev_proportion += allocation

        # Ensure at least one sample per client
        return self._ensure_minimum_allocation_per_client(client_indices)

    def _calculate_allocations(
        self, indices: torch.Tensor, proportions: torch.Tensor
    ) -> torch.Tensor:
        allocations = (proportions * len(indices)).round().int()
        while allocations.sum() < len(indices):
            allocations[torch.argmin(allocations)] += 1
        return allocations

    def _ensure_minimum_allocation_per_client(
        self, client_indices: List[List[int]]
    ) -> List[List[int]]:
        # Check and redistribute one sample to any empty client list
        max_length, max_subset_idx = max(
            (len(indices), idx) for idx, indices in enumerate(client_indices)
        )
        for i in range(self.num_clients):
            if len(client_indices[i]) == 0:
                if max_length > 1:
                    client_indices[i].append(client_indices[max_subset_idx].pop())
                    max_length -= 1
                else:
                    raise ValueError(
                        "Cannot ensure minimum allocation for each client."
                    )
        return client_indices
