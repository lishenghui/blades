import sys

import ray

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

# runtime parameters
run_params = {
    "model": CCTNet(),  # global model
    "server_optimizer": 'SGD',  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 50,  # number of steps per round
    "lr": 0.1,  # learning rate
}
simulator.run(**run_params)
