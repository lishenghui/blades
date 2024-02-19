from collections import defaultdict
from typing import DefaultDict, List, Optional, Dict

import numpy as np
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override

from fedlib.datasets import FLDataset
from fedlib.utils.types import NotProvided
from fedlib.algorithms import AlgorithmConfig
from fedlib.clients import ClientConfig
from fedlib.algorithms.fedavg.fedavg import Fedavg as FedavgAlgorithm
from fedlib.constants import CLIENT_UPDATE, GLOBAL_MODEL, NUM_GLOBAL_STEPS
from fedlib.algorithms.fedavg.fedavg import FedavgConfig as FedavgConfigBase

from blades.adversaries import Adversary, AdversaryConfig


class FedavgConfig(FedavgConfigBase):
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

    # def get_worker_group_config(self) -> WorkerGroupConfig:
    #     if not self._is_frozen:
    #         raise ValueError(
    #             "Cannot call `get_learner_group_config()` on an unfrozen "
    #             "AlgorithmConfig! Please call `freeze()` first."
    #         )

    #     config = (
    #         WorkerGroupConfig()
    #         .resources(
    #             num_remote_workers=self.num_remote_workers,
    #             num_cpus_per_worker=self.num_cpus_per_worker,
    #             num_gpus_per_worker=self.num_gpus_per_worker,
    #         )
    #         .task(task_spec=self.get_task_spec())
    #     )
    #     return config

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


class Fedavg(FedavgAlgorithm):
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

        self.adversary = self.config.get_adversary_config().build(
            self.client_manager.clients[: self.config.num_malicious_clients]
        )
        self.adversary.on_algorithm_start(self)

    def training_step(self):
        self.worker_group.sync_weights(self.server.get_global_model().state_dict())

        def local_training(worker, client):
            if isinstance(worker.dataset, FLDataset):
                dataset = worker.dataset.get_client_dataset(client.client_id)
            else:
                dataset = worker.dataset.get_train_loader(client.client_id)
            result = client.train_one_round(dataset)
            return result

        clients = self.client_manager.trainable_clients
        self.local_results = self.worker_group.foreach_execution(
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
