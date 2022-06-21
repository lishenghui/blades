import sys

import ray
import torch

sys.path.insert(0, '../..')
from blades.simulator import Simulator
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet

cifar10 = CIFAR10(num_clients=20, iid=True)  # built-in federated cifar10 dataset

# configuration parameters
conf_params = {
    "dataset": cifar10,
    "aggregator": "clippedclustering",  # defense: robust aggregation
    "num_byzantine": 0,  # number of byzantine input
    "use_cuda": False,
    "attack": "alie",  # attack strategy
    "attack_param": {"num_clients": 20,  # attacker parameters
                     "num_byzantine": 0},
    "num_actors": 4,  # number of training actors
    "gpu_per_actor": 0.0,
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=4)
simulator = Simulator(**conf_params)

model = CCTNet()
server_opt = torch.optim.Adam(model.parameters(), lr=0.01)
# runtime parameters
run_params = {
    "model": model,  # global model
    "server_optimizer": 'SGD',  # ,server_opt  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 7500,  # number of global rounds
    "local_steps": 1,  # number of s"client_lr": 0.1,  # learning rateteps per round
    "server_lr": 0.1,
    "client_lr": 1.0,  # learning rate
    "validate_interval": 10,
}
simulator.run(**run_params)
