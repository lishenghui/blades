import os

import ray
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

# from blades.utils.args import options
from blades.simulator import Simulator
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet

cifar10 = CIFAR10(num_clients=20, iid=True)  # built-in federated cifar10 dataset

# configuration parameters
conf_args = {
    "dataset": cifar10,
    "aggregator": "geomed",  # defense: robust aggregation
    # "aggregator": "clippedclustering",  # defense: robust aggregation
    "num_byzantine": 8,  # number of byzantine input
    "use_cuda": True,
    # "use_cuda": False,
    "attack": "ipm",  # attack strategy
    "attack_params": {
        # "num_clients": 20,  # attacker parameters
        # "num_byzantine": 8,
    },
    "num_actors": 20,  # number of training actors
    "gpu_per_actor": 0.19,
    # "gpu_per_actor": 0.19,
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=4)
# ray.init(local_mode=True)
simulator = Simulator(**conf_args)

model = CCTNet()
client_opt = torch.optim.Adam(model.parameters(), lr=0.1)
client_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    client_opt, milestones=[150, 300, 500], gamma=0.5
)
# runtime parameters
run_args = {
    "model": model,  # global model
    "server_optimizer": 'SGD',  # server_opt, server optimizer
    "client_optimizer": client_opt,  # client optimizer
    "loss": "crossentropy",  # loss funcstion
    "global_rounds": 600,  # number of global rounds
    "local_steps": 50,  # number of seps "client_lr": 0.1,  # learning rateteps per round
    "server_lr": 1.0,
    # "client_lr": 0.1,  # learning rate
    "validate_interval": 10,
    "client_lr_scheduler": client_lr_scheduler,
}
simulator.run(**run_args)
