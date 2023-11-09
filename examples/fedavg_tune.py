import ray
from ray import tune
from ray.tune.stopper import MaximumIterationStopper

from blades.algorithms.fedavg import FedavgConfig, Fedavg
from fllib.algorithms import AlgorithmConfig


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


class ExampleFedavg(Fedavg):
    def __init__(self, config=None, logger_creator=None, **kwargs):
        super().__init__(config, logger_creator, **kwargs)

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return ExampleFedavgConfig()


if __name__ == "__main__":
    ray.init()

    config_dict = (
        ExampleFedavgConfig()
        .resources(
            num_gpus_for_driver=0.5,
            num_cpus_for_driver=1,
            num_remote_workers=0,
            num_gpus_per_worker=0.5,
        )
        .to_dict()
    )
    print(config_dict)
    tune.run(
        ExampleFedavg,
        config=config_dict,
        stop=MaximumIterationStopper(100),
    )
