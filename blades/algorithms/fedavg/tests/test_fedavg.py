import unittest

import ray

import blades.algorithms.fedavg as fedavg

import torch
import torch.nn as nn
from fllib.models.catalog import ModelCatalog
import torch.optim as optim
from fllib.datasets.fldataset import FLDataset
from fllib.datasets.catalog import DatasetCatalog


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        return x


class SimpleDataset(FLDataset):
    def __init__(
        self,
        cache_name: str = "",
        iid=True,
        alpha=0.1,
        num_clients=1,
        seed=1,
        train_data=None,
        test_data=None,
        train_bs=3,
        num_classes=2,
    ) -> None:
        # Simple dataset
        features = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]).numpy()
        targets = torch.tensor([1.0, 0.0, 0.0]).numpy()
        self.num_classes = num_classes
        super().__init__(
            (features, targets),
            (features, targets),
            cache_name=cache_name,
            iid=iid,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
            train_data=train_data,
            test_data=test_data,
            train_bs=train_bs,
            is_image=False,
        )


NET_NAME = "simple_net"
DATASET_NAME = "simple_dataset"


class TestFedavg(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleDataset()
        self.net = SimpleNet()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.5)

    @classmethod
    def setUpClass(cls):
        ray.init()
        ModelCatalog.register_custom_model(NET_NAME, SimpleNet)
        DatasetCatalog.register_custom_dataset(DATASET_NAME, SimpleDataset)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_training(self):
        dataset = DatasetCatalog.get_dataset({"custom_dataset": DATASET_NAME})
        self.assertIsInstance(dataset, SimpleDataset)

        model = ModelCatalog.get_model({"custom_model": NET_NAME})
        self.assertIsInstance(model, SimpleNet)

        num_cpus_per_worker = 1
        num_gpus_per_worker = 1
        num_remote_workers = 1
        config = (
            fedavg.FedavgConfig()
            .resources(
                num_cpus_per_worker=num_cpus_per_worker,
                num_gpus_per_worker=num_gpus_per_worker,
                num_remote_workers=num_remote_workers,
            )
            .data(dataset_config={"custom_dataset": DATASET_NAME, "num_classes": 2})
            .training(global_model={"custom_model": NET_NAME})
        )
        algo = config.build()
        self.assertAlmostEqual(algo.config.num_cpus_per_worker, num_cpus_per_worker)
        self.assertAlmostEqual(algo.config.num_gpus_per_worker, num_gpus_per_worker)
        self.assertIsNotNone(algo)

        for _ in range(2000):
            results = algo.train()
            if results.get("acc_top_1"):
                print(results)
        self.assertIsInstance(results, dict)

    def test_basic(self):
        # net = SimpleNet()
        model = ModelCatalog.get_model({"custom_model": NET_NAME})
        self.assertIsInstance(model, SimpleNet)
        # num_cpus_per_worker = 1
        # num_gpus_per_worker = 1
        # num_remote_workers = 4
        # dataset = "cifar10"
        # num_clients = 4

        # config = (
        #     fedavg.FedavgConfig()
        #     .resources(
        #         num_cpus_per_worker=num_cpus_per_worker,
        #         num_gpus_per_worker=num_gpus_per_worker,
        #         num_remote_workers=num_remote_workers,
        #     )
        #     .data(dataset=dataset, num_clients=num_clients)
        #     .training(global_model="resnet18")
        #     # .adversary(num_malicious_clients=2, attack_type="label_flipping")
        # )
        # algo = config.build()
        # self.assertAlmostEqual(algo.config.num_cpus_per_worker, num_cpus_per_worker)
        # self.assertAlmostEqual(algo.config.num_gpus_per_worker, num_gpus_per_worker)
        # self.assertIsNotNone(algo)

        # for _ in range(20):
        #     results = algo.train()
        # self.assertIsInstance(results, dict)
        # self.assertTrue(False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
