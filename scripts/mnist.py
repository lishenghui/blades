import os

import ray
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

from args import options
from blades.simulator import Simulator
from blades.datasets import MNIST
from blades.models.mnist import MLP

args = options
# if not ray.is_initialized():
#     print("initializing ray")
#     ray.init(include_dashboard=False, num_gpus=args.num_gpus)
# ray.init(include_dashboard=False, num_gpus=args.num_gpus)
# else:
    # ray.init(address='auto')
ray.init(address='auto')

if not os.path.exists(options.log_dir):
    os.makedirs(options.log_dir)

mnist = MNIST(num_clients=options.num_clients, iid=True, seed=0)  # built-in federated mnist dataset

# configuration parameters
conf_args = {
    "dataset": mnist,
    "aggregator": options.agg,  # defense: robust aggregation
    "aggregator_kws": options.agg_args[options.agg],
    "num_byzantine": options.num_byzantine,  # number of byzantine input
    "use_cuda": True,
    "attack": options.attack,  # attack strategy
    "attack_kws": options.attack_args[options.attack],
    "num_actors": 5,  # number of training actors
    "gpu_per_actor": 0.2,
    "log_path": options.log_dir,
    "seed": options.seed,  # reproducibility
}

simulator = Simulator(**conf_args)

model = MLP()
client_opt = torch.optim.SGD(model.parameters(), lr=1.0)
# client_opt = torch.optim.SGD(model.parameters(), lr=0.1)
client_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    client_opt, milestones=[150, 300, 500], gamma=0.5
)
# runtime parameters
run_args = {
    "model": model,  # global model
    "server_optimizer": 'SGD',  # server_opt, server optimizer
    "client_optimizer": client_opt,  # client optimizer
    "loss": "crossentropy",  # loss funcstion
    "global_rounds": options.global_round,  # number of global rounds
    "local_steps": options.local_round,  # number of seps "client_lr": 0.1,  # learning rateteps per round
    "server_lr": 0.1,
    # "server_lr": 1.0,
    "validate_interval": 20,
    # "client_lr_scheduler": client_lr_scheduler,
    # "client_lr_scheduler": client_lr_scheduler,
}
simulator.run(**run_args)
