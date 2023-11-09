import copy
import unittest

import ray
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from blades.adversaries import NoiseAdversary
from blades.algorithms.fedavg import FedavgConfig
from fllib.datasets import DatasetCatalog
from fllib.datasets.tests.toy_dataset import ToyFLDataset


class TestAdaptiveAdversary(unittest.TestCase):
    def setUp(self):
        DatasetCatalog.register_custom_dataset("simple", ToyFLDataset)
        model = torch.nn.Linear(2, 2)
        self.alg = (
            FedavgConfig()
            .resources(num_remote_workers=2, num_gpus_per_worker=0)
            .data(
                dataset_config={
                    "custom_dataset": "simple",
                },
            )
            .training(
                global_model=model,
                server_config={"lr": 0.1, "aggregator": {"type": "Mean"}},
            )
            .adversary(
                num_malicious_clients=1,
                adversary_config={"type": NoiseAdversary},
            )
            .build()
        )
        self.global_dataset = DatasetCatalog.get_dataset(
            {
                "custom_dataset": "simple",
            },
        )

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_on_local_round_end(self):
        train_set, _ = self.global_dataset.to_torch_datasets()

        for _ in range(5):
            for data, target in DataLoader(dataset=train_set, batch_size=3):
                model = copy.deepcopy(self.alg.server.get_global_model())
                opt = torch.optim.SGD(model.parameters(), lr=0.1)
                model.train()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                opt.step()
                break

            self.alg.training_step()
            updated_model = copy.deepcopy(self.alg.server.get_global_model())

            self.assertFalse(torch.allclose(model.weight, updated_model.weight))


if __name__ == "__main__":
    unittest.main()
