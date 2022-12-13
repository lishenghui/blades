"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""

import ray
import torch

from blades.core.simulator import Simulator
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet

# built-in federated cifar10 dataset
cifar10 = CIFAR10(num_clients=20, iid=True)


class Median:
    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2


# configuration parameters
conf_params = {
    "dataset": cifar10,
    "aggregator": Median(),  # defense: robust aggregation
    "num_actors": 20,  # number of training actors
    "gpu_per_actor": 0.19,
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=4)
simulator = Simulator(**conf_params)

# runtime parameters
run_params = {
    "global_model": CCTNet(),  # global global_model
    "server_optimizer": "SGD",  # server optimizer
    "client_optimizer": "SGD",  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 20,  # number of steps per round
    "client_lr": 0.1,  # learning rate
}
simulator.run(**run_params)
