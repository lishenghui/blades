import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from fllib.datasets.splitters import DirichletSplitter


# Mock Dataset
class MockDataset(Dataset):
    def __init__(self, size, num_classes):
        self.size = size
        self.num_classes = num_classes
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.targets[idx]


class TestDirichletSplitter(unittest.TestCase):
    def test_split(self):
        dataset_size = 100
        num_classes = 10
        num_clients = 5
        alpha = 0.1

        # Create mock dataset
        dataset = MockDataset(dataset_size, num_classes)

        # Initialize Dirichlet splitter
        splitter = DirichletSplitter(num_clients=num_clients, alpha=alpha)

        # Split the dataset
        subsets = splitter.split_dataset(dataset)

        # Check that we have the correct number of subsets
        self.assertEqual(len(subsets), num_clients)

        # Check that each client has at least one sample
        for subset in subsets:
            self.assertGreater(len(subset), 0)

        # Check that all samples are allocated
        all_indices = [idx for subset in subsets for idx in subset.indices]
        self.assertEqual(len(set(all_indices)), dataset_size)

    def test_class_distribution(self):
        dataset_size = 1000  # 增加数据集大小以获得更稳定的统计结果
        num_classes = 10
        num_clients = 5
        alpha = 100.0  # 使用一个更中等的alpha值以期望更均匀的分布

        # Create mock dataset
        dataset = MockDataset(dataset_size, num_classes)

        # Initialize Dirichlet splitter
        splitter = DirichletSplitter(num_clients=num_clients, alpha=alpha)

        # Split the dataset
        subsets = splitter.split_dataset(dataset)

        # Collect class distributions
        class_distributions = {i: [] for i in range(num_classes)}
        for i in range(num_classes):
            for subset in subsets:
                class_count = sum([1 for idx in subset.indices if dataset[idx] == i])
                class_distributions[i].append(class_count)
        # Check the distribution for each class across clients
        # for class_id, distributions in class_distributions.items():
        # Here we could use a statistical test or simply check if no client is missing
        # this class self.assertTrue(all(distribution > 0 for distribution
        # in distributions))

        # The code above only checks for non-zero distributions.
        # For more detailed testing, you might want to use actual statistical methods.

    def test_effect_of_alpha(self):
        dataset_size = 1000  # Larger dataset size for more stable statistics
        num_classes = 10
        num_clients = 5
        alphas = [0.001, 1.0, 100.0]  # Different alpha values
        prev_std_devs = np.inf
        for alpha in alphas:
            dataset = MockDataset(dataset_size, num_classes)
            splitter = DirichletSplitter(num_clients=num_clients, alpha=alpha)
            subsets = splitter.split_dataset(dataset)

            # Calculate the distribution of each class for each client
            class_distributions = {i: [] for i in range(num_classes)}
            for i in range(num_classes):
                for subset in subsets:
                    class_count = sum(1 for idx in subset.indices if dataset[idx] == i)
                    class_distributions[i].append(class_count)

            # Calculate the standard deviation of distributions for each class
            std_devs = np.std(list(class_distributions.values()), axis=1).mean()
            # Standard deviation should decrease as alpha increases
            if alpha > 0.01:
                self.assertTrue(std_devs < prev_std_devs)

            prev_std_devs = std_devs

    def test_splits(self):
        dataset_size = 100
        num_classes = 10
        num_clients = 5
        alpha = 0.1

        # Create mock dataset
        dataset = MockDataset(dataset_size, num_classes)
        dataset_test = MockDataset(dataset_size, num_classes)

        # Initialize Dirichlet splitter
        splitter = DirichletSplitter(
            num_clients=num_clients, alpha=alpha, same_proportions=True
        )

        # Split the dataset
        subsets = splitter.split_datasets(dataset, dataset_test)[0]

        # Check that we have the correct number of subsets
        self.assertEqual(len(subsets), num_clients)

        # Check that each client has at least one sample
        for subset in subsets:
            self.assertGreater(len(subset), 0)

        # Check that all samples are allocated
        # all_indices = [idx for subset in subsets for idx in subsets]
        # self.assertEqual(len(set(all_indices)), dataset_size)


if __name__ == "__main__":
    unittest.main()
