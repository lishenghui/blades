from collections import defaultdict
from typing import DefaultDict, List, Optional, Dict

import numpy as np
import ray
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.util.timer import _Timer

from blades.adversaries import Adversary, AdversaryConfig
from fllib.algorithms import Algorithm, AlgorithmConfig
from fllib.clients import ClientConfig
from fllib.constants import CLIENT_UPDATE, GLOBAL_MODEL, NUM_GLOBAL_STEPS
from fllib.core.execution.worker_group import WorkerGroup
from fllib.core.execution.worker_group_config import WorkerGroupConfig
from fllib.datasets import DatasetCatalog
from fllib.datasets.dataset import FLDataset
from fllib.types import NotProvided
from fllib.types import PartialAlgorithmConfigDict


class FedavgConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        """Initializes a FedavgConfig instance."""
        super().__init__(algo_class=algo_class or Fedavg)

        self.adversary_config = {}

        self.num_malicious_clients = 0

    def adversary(
        self,
        *,
        num_malicious_clients: Optional[int] = NotProvided,
        adversary_config: Optional[Dict] = NotProvided,
    ):
        if adversary_config is not NotProvided:
            deep_update(
                self.adversary_config,
                adversary_config,
                True,
            )
        if num_malicious_clients is not NotProvided:
            self.num_malicious_clients = num_malicious_clients
        return self

    def get_adversary_config(self) -> AdversaryConfig:
        if not self._is_frozen:
            raise ValueError(
                "Cannot call `get_learner_group_config()` on an unfrozen "
                "AlgorithmConfig! Please call `freeze()` first."
            )
        config = AdversaryConfig(
            adversary_cls=self.adversary_config.get("type", Adversary), config=self
        ).update_from_dict(self.adversary_config)
        return config

    def get_client_config(self) -> ClientConfig:
        config = ClientConfig(class_specifier="blades.clients.Client").update_from_dict(
            self.client_config
        )
        return config

    def get_worker_group_config(self) -> WorkerGroupConfig:
        if not self._is_frozen:
            raise ValueError(
                "Cannot call `get_learner_group_config()` on an unfrozen "
                "AlgorithmConfig! Please call `freeze()` first."
            )

        config = (
            WorkerGroupConfig()
            .resources(
                num_remote_workers=self.num_remote_workers,
                num_cpus_per_worker=self.num_cpus_per_worker,
                num_gpus_per_worker=self.num_gpus_per_worker,
            )
            .task(task_spec=self.get_task_spec())
        )
        return config

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()

        # This condition is only for DnC and Trimmedmean aggregators.
        if (
            self.server_config.get("aggregator", None) is not None
            and self.server_config.get("aggregator").get("type") is not None
            # and "num_byzantine" not in self.server_config["aggregator"]["type"]
            and (
                "DnC" == self.server_config["aggregator"]["type"]
                or "Trimmedmean" in self.server_config["aggregator"]["type"]
                or "Multikrum" in self.server_config["aggregator"]["type"]
            )
        ):
            self.server_config["aggregator"][
                "num_byzantine"
            ] = self.num_malicious_clients

        # Check whether the number of malicious clients makes sense.
        if self.num_malicious_clients > self.num_clients:
            raise ValueError(
                "`num_malicious_clients` must be smaller than or equal "
                "`num_clients`! Simulation makes no sense otherwise."
            )


class Fedavg(Algorithm):
    """Federated Averaging Algorithm."""

    def __init__(self, config=None, logger_creator=None, **kwargs):
        self._client_actors_affinity: DefaultDict[int, List[int]] = defaultdict(list)
        self.local_results = []
        super().__init__(config, logger_creator, **kwargs)

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return FedavgConfig()

    def setup(self, config: AlgorithmConfig):
        super().setup(config)
        # Set up our config: Merge the user-supplied config dict (which could
        # be a partial config dict) with the class' default.
        if not isinstance(config, AlgorithmConfig):
            assert isinstance(config, PartialAlgorithmConfigDict)
            config_obj = self.get_default_config()
            if not isinstance(config_obj, AlgorithmConfig):
                assert isinstance(config, PartialAlgorithmConfigDict)
                config_obj = AlgorithmConfig().from_dict(config_obj)
            config_obj.update_from_dict(config)
            self.config = config_obj

        # Set Algorithm's seed.
        update_global_seed_if_necessary("torch", self.config.random_seed)

        server_device = "cuda" if self.config.num_gpus_for_driver > 0 else "cpu"
        self.server = self.config.get_server_config().build(server_device)
        self.worker_group = self._setup_worker_group()

        self._setup_dataset()
        self.client_manager = self.client_manager_cls(
            self._dataset.client_ids,
            self._dataset.train_client_ids,
            self._dataset.test_client_ids,
            client_config=self.config.get_client_config(),
        )

        # Metrics-related properties.
        self._timers = defaultdict(_Timer)
        self._counters = defaultdict(int)
        self.global_vars = defaultdict(lambda: defaultdict(list))

        clients = self.client_manager.clients
        if self.worker_group.workers:
            affinity_actors = [
                self._client_actors_affinity[client.client_id] for client in clients
            ]
            self.local_results = self.worker_group.execute_with_actor_pool(
                lambda _, client: client.setup(), clients, affinity_actors
            )
        else:
            self.local_results = self.worker_group.local_execute(
                lambda _, client: client.setup(), clients
            )

        self.adversary = self.config.get_adversary_config().build(
            self.client_manager.clients[: self.config.num_malicious_clients]
        )
        self.adversary.on_algorithm_start(self)

    def _setup_worker_group(self) -> WorkerGroup:
        worker_group_config = self.config.get_worker_group_config()
        worker_group = worker_group_config.build()
        return worker_group

    def _setup_dataset(self):
        self._dataset = DatasetCatalog.get_dataset(self.config.dataset_config)

        if self.worker_group.workers:

            def setup_dataset(subset):
                def bind_dataset(worker):
                    setattr(worker, "dataset", subset)
                    return ray.get_runtime_context().get_actor_id(), subset.client_ids

                return bind_dataset

            subdatasets = self._dataset.split(len(self.worker_group.workers))
            results = self.worker_group.foreach_actor(
                [setup_dataset(subset) for subset in subdatasets]
            )
            results = [result.get() for result in results.ignore_errors()]

            for actor, clients in results:
                for client in clients:
                    self._client_actors_affinity[client].append(actor)
        else:
            self.worker_group.local_worker().dataset = self._dataset

    def training_step(self):
        self.worker_group.sync_state(
            GLOBAL_MODEL, self.server.get_global_model().state_dict()
        )

        def local_training(worker, client):
            if isinstance(worker.dataset, FLDataset):
                dataset = worker.dataset.get_client_dataset(client.client_id)
            else:
                dataset = worker.dataset.get_train_loader(client.client_id)
            return client.train_one_round(dataset)

        clients = self.client_manager.trainable_clients
        if self.worker_group.workers:
            affinity_actors = [
                self._client_actors_affinity[client.client_id] for client in clients
            ]
            self.local_results = self.worker_group.execute_with_actor_pool(
                local_training, clients, affinity_actors
            )
        else:
            self.local_results = self.worker_group.local_execute(
                local_training, clients
            )

        self.adversary.on_local_round_end(self)

        updates = [result.pop(CLIENT_UPDATE, None) for result in self.local_results]

        losses = []
        for result in self.local_results:
            client = self.client_manager.get_client_by_id(result["id"])
            if not client.is_malicious:
                loss = result.pop("avg_loss")
                losses.append(loss)

        self._counters[NUM_GLOBAL_STEPS] += 1
        # train_results
        global_vars = {
            "timestep": self._counters[NUM_GLOBAL_STEPS],
        }
        results = {"train_loss": np.mean(losses)}
        server_return = self.server.step(updates, global_vars)
        results.update(server_return)

        return results

    def evaluate(self):
        self.worker_group.sync_state(
            GLOBAL_MODEL, self.server.get_global_model().state_dict()
        )
        clients = self.client_manager.testable_clients

        def validate_func(worker, client):
            # test_loader = worker.dataset.get_test_loader(client.client_id)
            if isinstance(worker.dataset, FLDataset):
                test_loader = worker.dataset.get_client_dataset(
                    client.client_id
                ).get_test_loader()
            else:
                test_loader = worker.dataset.get_test_loader(client.client_id)
            return client.evaluate(test_loader)

        if self.worker_group.workers:
            affinity_actors = [
                self._client_actors_affinity[client.client_id] for client in clients
            ]

            evaluate_results = self.worker_group.execute_with_actor_pool(
                validate_func, clients, affinity_actors
            )
        else:
            evaluate_results = self.worker_group.local_execute(validate_func, clients)
        results = {
            "ce_loss": np.average(
                [metric["ce_loss"] for metric in evaluate_results],
                weights=[metric["length"] for metric in evaluate_results],
            ),
            "acc_top_1": np.average(
                [metric["acc_top_1"].cpu() for metric in evaluate_results],
                weights=[metric["length"] for metric in evaluate_results],
            ),
        }

        return results

    def save_checkpoint(self, checkpoint_dir: str) -> Dict | None:
        pass
