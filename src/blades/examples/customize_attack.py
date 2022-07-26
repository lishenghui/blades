"""
Customization of attack strategy
=================================

"""


import ray

from blades.client import ByzantineClient
from blades.datasets import MNIST
from blades.models.mnist import MLP
from blades.simulator import Simulator

mnist = MNIST(data_root="./data", train_bs=32, num_clients=10)  # built-in federated MNIST dataset


class MaliciousClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def omniscient_callback(self, simulator):
        updates = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                updates.append(w.get_update())
        self.save_update(-100 * (sum(updates)) / len(updates))


# configuration parameters
conf_params = {
    "dataset": mnist,
    "aggregator": "geomed",  # defense: robust aggregation
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0, local_mode=True)
simulator = Simulator(**conf_params)


# %%
# Register attacks in the simulator.

attackers = [MaliciousClient() for _ in range(5)]
simulator.register_attackers(attackers)


# %%
# Configure run time parameters and run the experiment.
run_params = {
    "model": MLP(),  # global model
    "server_optimizer": 'SGD',  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 50,  # number of steps per round
    "client_lr": 0.1,  # learning rate
}
simulator.run(**run_params)
