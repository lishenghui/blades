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
    "aggregator": "krum",  # defense: robust aggregation
    "num_byzantine": 5,  # number of byzantine clients
    "use_cuda": True,
    "attack": "noise",  # attack strategy
    # "attack_para":{"n": 20, # attacker parameters
    #                "m": 5},
    "num_actors": 20,  # number of training actors
    "gpu_per_actor": 0.19,
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=4)
simulator = Simulator(**conf_params)


model = CCTNet()
server_opt = torch.optim.Adam(model.parameters(), lr=0.01)
# runtime parameters
run_params = {
    "model": model,  # global model
    "server_optimizer": server_opt, #'SGD',  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 10,  # number of s"client_lr": 0.1,  # learning rateteps per round
    
}
simulator.run(**run_params)
