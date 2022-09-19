"""
A example for CIFAR10
===========================

"""


import ray

from blades.simulator import Simulator
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet

cifar10 = CIFAR10(num_clients=20, train_bs=32, seed=3)  # built-in federated cifar10 dataset

# configuration parameters
conf_params = {
    "dataset": cifar10,
    "aggregator": "median",  # defense: robust aggregation
    "num_byzantine": 5,  # number of byzantine input
    "use_cuda": True,
    "attack": "noise",  # attack strategy
    "attack_kws": {
        # "num_clients": 20,
        # "num_byzantine": 8,
        },
    "num_actors": 20,  # number of training actors
    "gpu_per_actor": 0.19,
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=4)
simulator = Simulator(**conf_params)

model = CCTNet()

# runtime parameters
run_params = {
    "model": model,  # global model
    "server_optimizer": 'SGD',  # ,server_opt  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 50,  # number of global rounds
    "local_steps": 50,  # number of steps per round
    "server_lr": 1.0,
    "client_lr": 0.1,  # learning rate
}
simulator.run(**run_params)
