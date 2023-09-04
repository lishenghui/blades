import unittest

import torch

from fllib.datasets.fldataset import FLDataset
from fllib.models.catalog import ModelCatalog
from fllib.datasets.catalog import DatasetCatalog
from fllib.tasks import TaskSpec
from fllib.clients import ClientConfig
from blades.algorithms.fedavg import FedavgConfig
from fllib.core.execution.worker_group_config import WorkerGroupConfig


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
        train_bs=1,
    ) -> None:
        # Simple dataset
        features = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).numpy()
        targets = torch.tensor([[0], [1], [1], [0]]).numpy()
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


def setup_task():
    # Define a custom binary classification model
    def simple_model():
        return torch.nn.Sequential(torch.nn.Linear(2, 2))

    # Register the custom model with the model catalog
    ModelCatalog.register_custom_model("SimpleModel", simple_model)

    alg_config = (
        FedavgConfig()
        .data(num_clients=2, dataset_config={"type": "mnist"})
        .training(global_model={"custom_model": "SimpleModel"})
    )
    return TaskSpec(task_class="fllib.tasks.MNIST", alg_config=alg_config)


def setup_worker_group():
    wg = (
        WorkerGroupConfig()
        .resources(num_remote_workers=1, num_cpus_per_worker=1, num_gpus_per_worker=1)
        .task(setup_task())
        .build()
    )
    return wg


class TestClientIntegration(unittest.TestCase):
    def setUp(self):
        DatasetCatalog.register_custom_dataset("simple", SimpleDataset)
        self.dataset = DatasetCatalog.get_dataset({"custom_dataset": "simple"})
        self.client_config = ClientConfig(
            class_specifier="fllib.clients.Client"
        ).client_id("0")
        self.client = self.client_config.build()
        self.wg = setup_worker_group()

    def test_train_one_round_integration(self):
        # Create a mock data reader that returns the same data every time
        self.wg.execute_with_actor_pool(
            lambda worker, client: setattr(
                worker,
                "dataset",
                DatasetCatalog.get_dataset({"custom_dataset": "simple"}),
            ),
            # lambda worker, client: setattr(worker, "dataset", self.dataset),
            values=[self.client],
        )
        # Train one round using the client
        result = self.wg.execute_with_actor_pool(
            lambda worker, client: client.train_one_round(
                worker.dataset.get_train_loader(client.client_id)
            ),
            values=[self.client],
        )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
