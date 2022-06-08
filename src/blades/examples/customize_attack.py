import ray

from blades.client import ByzantineClient
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet
from blades.simulator import Simulator


cifar10 = CIFAR10(num_clients=20, iid=True)  # built-in federated cifar10 dataset


class MaliciousClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super(ByzantineClient).__init__(*args, **kwargs)
    
    def omniscient_callback(self, simulator):
        updates = []
        for w in simulator._clients:
            is_byzantine = w.get_is_byzantine()
            if not is_byzantine:
                updates.append(w.get_update())
        self._save_update(-100 * (sum(updates)) / len(updates))


# configuration parameters
conf_params = {
    "dataset": cifar10,
    "aggregator": "mean",  # defense: robust aggregation
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0)
simulator = Simulator(**conf_params)

attackers = [MaliciousClient() for _ in range(5)]
simulator.register_attackers(attackers)

# runtime parameters
run_params = {
    "model": CCTNet(),  # global model
    "server_optimizer": 'SGD',  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 2,  # number of steps per round
    "client_lr": 0.1,  # learning rate
}
simulator.run(**run_params)
