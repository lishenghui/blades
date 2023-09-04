import unittest

import torch

from fllib.blades.adversaries import (
    ALIEAdversary,
    IPMAdversary,
    SignFlipAdversary,
    LabelFlipAdversary,
    AdaptiveAdversary,
)
from fllib.blades.algorithms.fedavg import FedavgConfig
from fllib.datasets.catalog import DatasetCatalog

from .simple_dataset import SimpleDataset


class TestAdversary(unittest.TestCase):
    def setUp(self):
        DatasetCatalog.register_custom_dataset("simple", SimpleDataset)

        self.algorithm = (
            FedavgConfig()
            .data(num_clients=2, dataset_config={"custom_dataset": "simple"})
            .training(global_model=torch.nn.Linear(5, 5))
            .adversary(
                num_malicious_clients=1,
                adversary_config={"type": "fllib.blades.adversaries.AdaptiveAdversary"},
            )
            .build()
        )
        self.adversary = self.algorithm.adversary

    def test_on_algorithm_start(self):
        for client in self.adversary.clients:
            self.assertTrue(client.is_malicious)

    def test_config(self):
        all_advs = [
            ALIEAdversary,
            IPMAdversary,
            SignFlipAdversary,
            LabelFlipAdversary,
            AdaptiveAdversary,
        ]

        for adv_cls in all_advs:
            config = (
                FedavgConfig()
                .data(
                    num_clients=1,
                    dataset_config={
                        "custom_dataset": "simple",
                        "num_classes": 2,
                        # "custom_dataset_config": {"num_classes": 2},
                    },
                )
                .training(global_model=torch.nn.Linear(5, 5))
            )
            if adv_cls == IPMAdversary:
                adv = (
                    config.adversary(adversary_config={"type": adv_cls, "scale": 0.1})
                    .build()
                    .adversary
                )
            else:
                adv = (
                    config.adversary(adversary_config={"type": adv_cls})
                    .build()
                    .adversary
                )
            self.assertIsInstance(adv, adv_cls)


if __name__ == "__main__":
    unittest.main()
