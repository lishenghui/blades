"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""

import ray

from blades.datasets import MNIST
from blades.models.mnist import MLP
from blades.simulator import Simulator

mnist = MNIST(data_root="./data", train_bs=32, num_clients=10)  # built-in federated MNIST dataset

# configuration parameters
conf_params = {
    "dataset": mnist,
    "aggregator": "fltrust",  # aggregation
    # "aggregator_params": {"num_clients": 10,  # attacker parameters
    #                 "num_byzantine": 3},
    "num_byzantine": 3,  # number of Byzantine input
    "attack": "alie",  # attack strategy
    "attack_params": {"num_clients": 10,  # attacker parameters
                     "num_byzantine": 3},
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0, local_mode=True)
simulator = Simulator(**conf_params)
trusted_id = list(simulator.get_clients().keys())[-1]
simulator.set_trusted_clients([trusted_id])
model = MLP()
# runtime parameters
run_params = {
    "model": model,  # global model
    "server_optimizer": 'SGD',  # ,server_opt  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 2,  # number of steps per round
    "server_lr": 1,
    "client_lr": 0.1,  # learning rate
}
simulator.run(**run_params)
