from typing import List

import torch
from torch.utils.data import Dataset, Subset

from .dataset_splitter import DatasetSplitter


class IIDSplitter(DatasetSplitter):
    def split_dataset(self, dataset: Dataset) -> List[Subset]:
        # Shuffle the dataset indices using PyTorch
        indices = torch.randperm(len(dataset)).tolist()

        # Calculate the size of each split, allowing for uneven splits
        split_size = len(dataset) // self.num_clients
        # Calculate the number of datasets that will have an extra sample to account
        # for remainders
        remainder = len(dataset) % self.num_clients

        # Generate the subsets
        subsets = []
        start_idx = 0
        for i in range(self.num_clients):
            end_idx = start_idx + split_size + (1 if i < remainder else 0)
            subsets.append(Subset(dataset, indices[start_idx:end_idx]))
            start_idx = end_idx

        return subsets

    def split_datasets(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> tuple[list[Subset], list[Subset]]:
        # Use the split_dataset method to split both the training and testing datasets
        train_subsets = self.split_dataset(train_dataset)
        test_subsets = self.split_dataset(test_dataset)

        return train_subsets, test_subsets
