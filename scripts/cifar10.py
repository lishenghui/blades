import sys

import ray
import torch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

from args import options
from blades.simulator import Simulator
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet

args = options
# if not ray.is_initialized():
ray.init(include_dashboard=False, num_gpus=args.num_gpus)
# ray.init(include_dashboard=False, num_gpus=args.num_gpus, local_mode=True)

if not os.path.exists(options.log_dir):
    os.makedirs(options.log_dir)

cifar10 = CIFAR10(num_clients=20, iid=True)  # built-in federated cifar10 dataset

# configuration parameters
conf_args = {
    "dataset": cifar10,
    "aggregator": options.agg,  # defense: robust aggregation
    "aggregator_kws": options.agg_args[options.agg],
    "num_byzantine": options.num_byzantine,  # number of byzantine input
    "use_cuda": True,
    "attack": options.attack,  # attack strategy
    "attack_kws": options.attack_args[options.attack],
    "num_actors": 20,  # number of training actors
    "gpu_per_actor": 0.19,
    "log_path": options.log_dir,
    "seed": options.seed,  # reproducibility
}

simulator = Simulator(**conf_args)

model = CCTNet()
client_opt = torch.optim.Adam(model.parameters(), lr=0.1)
client_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        client_opt, milestones=[150, 300, 500], gamma=0.5
    )
# runtime parameters
run_args = {
    "model": model,  # global model
    "server_optimizer": 'SGD', #server_opt, server optimizer
    "client_optimizer": client_opt,  # client optimizer
    "loss": "crossentropy",  # loss funcstion
    "global_rounds": options.global_round,  # number of global rounds
    "local_steps": options.local_round,  # number of seps "client_lr": 0.1,  # learning rateteps per round
    "server_lr": 1.0,
    # "client_lr": 0.1,  # learning rate
    "validate_interval": 10,
    "client_lr_scheduler": client_lr_scheduler,
}
simulator.run(**run_args)
