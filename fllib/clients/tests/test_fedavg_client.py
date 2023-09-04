import unittest
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from fllib.clients.fedavg_client import FedavgClientConfig, FedavgClient


class TestFedavgClient(unittest.TestCase):
    def setUp(self):
        self.train_batch_size = 10
        self.config = (
            FedavgClientConfig()
            .client_id("client1")
            .dataset("mnist", train_batch_size=self.train_batch_size, eva_batch_size=64)
            .model("mlp", loss="cross_entropy")
            .optimizer("sgd", opt_kargs={"lr": 0.1234})
        )
        self.client = FedavgClient(self.config)

    def test_init(self):
        self.assertIsInstance(self.client, FedavgClient)
        # self.assertIsInstance(self.client.model, torch.nn.Module)
        # self.assertIsInstance(self.client.optimizer, torch.optim.SGD)
        # self.assertIsInstance(self.client.loss, torch.nn.CrossEntropyLoss)
        # self.assertAlmostEqual(self.client.optimizer.param_groups[0]["lr"], 0.1234)

    def test_train_one_batch(self):
        pass
        # data_reader = self.generate_dummy_data()
        # data, target = next(data_reader)
        # loss = self.client._train_one_batch(data, target)
        # self.assertIsInstance(loss, float)

    def test_train_one_round(self):
        pass
        # data_reader = self.generate_dummy_data()
        # result = self.client.train_one_round(data_reader)
        # print(result)
        # self.assertIsInstance(result, dict)

    def generate_dummy_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        class FakeCIFAR10(Dataset):
            def __init__(self, length):
                self.length = length
                self.data = torch.randn(length, 3, 28, 28)
                self.targets = torch.randint(0, 10, (length,))

            def __getitem__(self, index):
                return self.data[index], self.targets[index]

            def __len__(self):
                return self.length

        # Create a fake cifar10 dataset with length 100
        fake_cifar10_dataset = FakeCIFAR10(100)

        # Define a DataLoader
        dataloader = DataLoader(
            fake_cifar10_dataset, batch_size=self.train_batch_size, shuffle=True
        )
        return iter(dataloader)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
