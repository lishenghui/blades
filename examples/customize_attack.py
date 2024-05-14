"""
Customize Attack
==================

"""


import ray
from ray import tune
from ray.tune.stopper import MaximumIterationStopper

from blades.algorithms.fedavg import FedavgConfig, Fedavg
from fedlib.trainers import TrainerConfig


from fedlib.trainers import Trainer
from fedlib.clients import ClientCallback
from blades.adversaries import Adversary


class LabelFlipAdversary(Adversary):
    def on_trainer_init(self, trainer: Trainer):
        class LabelFlipCallback(ClientCallback):
            def on_train_batch_begin(self, data, target):
                return data, 10 - 1 - target

        for client in self.clients:
            client.to_malicious(callbacks_cls=LabelFlipCallback, local_training=True)


class ExampleFedavgConfig(FedavgConfig):
    def __init__(self, algo_class=None):
        """Initializes a FedavgConfig instance."""
        super().__init__(algo_class=algo_class or ExampleFedavg)

        self.dataset_config = {
            "type": "FashionMNIST",
            "num_clients": 10,
            "train_batch_size": 32,
        }
        self.global_model = "cnn"
        self.num_malicious_clients = 1
        self.adversary_config = {"type": LabelFlipAdversary}


class ExampleFedavg(Fedavg):
    @classmethod
    def get_default_config(cls) -> TrainerConfig:
        return ExampleFedavgConfig()


if __name__ == "__main__":
    ray.init()

    config_dict = (
        ExampleFedavgConfig()
        .resources(
            num_gpus_for_driver=0.0,
            num_cpus_for_driver=1,
            num_remote_workers=0,
            num_gpus_per_worker=0.0,
        )
        .to_dict()
    )
    print(config_dict)
    tune.run(
        ExampleFedavg,
        config=config_dict,
        stop=MaximumIterationStopper(100),
    )
