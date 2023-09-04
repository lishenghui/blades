import unittest

import ray

import fllib.algorithms.fedavg as fedavg


class TestBC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_init(self):
        num_cpus_per_worker = 1
        num_gpus_per_worker = 1
        num_remote_workers = 4
        dataset = "cifar10"
        num_clients = 20

        config = (
            fedavg.FedavgConfig()
            .resources(
                num_cpus_per_worker=num_cpus_per_worker,
                num_gpus_per_worker=num_gpus_per_worker,
                num_remote_workers=num_remote_workers,
            )
            .data(dataset=dataset, num_clients=num_clients)
            .adversary(num_malicious_clients=2, attack_type="label-flipping")
        )
        algo = config.build()
        self.assertAlmostEqual(algo.config.num_cpus_per_worker, num_cpus_per_worker)
        self.assertAlmostEqual(algo.config.num_gpus_per_worker, num_gpus_per_worker)
        self.assertIsNotNone(algo)

        for _ in range(50):
            results = algo.train()
        self.assertIsInstance(results, dict)
        # self.assertTrue(False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
