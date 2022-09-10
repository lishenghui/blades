import os

import ray
import torch

from blades.models.mnist import MLP
from args import options
from blades.simulator import Simulator
from blades.datasets import CIFAR10
from blades.datasets import MNIST
import math
from blades.models.cifar10 import CCTNet


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

args = options
ray.init(address='auto')

if not os.path.exists(options.log_dir):
    os.makedirs(options.log_dir)

data_root = "./data"
cache_name = options.dataset + "_" + options.algorithm + ("_noniid" if not options.noniid else "") + f"_{str(options.num_clients)}_{str(options.seed)}"
if options.dataset == 'cifar10':
    dataset = CIFAR10(data_root=data_root, cache_name=cache_name, train_bs=options.batch_size, num_clients=options.num_clients, iid=not options.noniid, seed=0)  # built-in federated cifar10 dataset
    model = CCTNet()
elif options.dataset == 'mnist':
    dataset = MNIST(data_root=data_root, cache_name=cache_name, train_bs=options.batch_size, num_clients=options.num_clients, iid=not options.noniid, seed=0)  # built-in federated cifar10 dataset
    model = MLP()
else:
    raise NotImplementedError

if options.gpu_per_actor > 0.0:
    model = model.to("cuda")


privacy_factor = args.privacy_sensitivity * math.sqrt(2 * math.log(1.25 / args.privacy_delta)) / args.privacy_epsilon
   
# configuration parameters
conf_args = {
    "dataset": dataset,
    "aggregator": options.agg,  # defense: robust aggregation
    "aggregator_kws": options.agg_args[options.agg],
    "num_byzantine": options.num_byzantine,  # number of byzantine input
    "use_cuda": options.gpu_per_actor > 0.0,
    "attack": options.attack,  # attack strategy
    "attack_kws": options.attack_args[options.attack],
    "adversary_kws": options.adversary_args,
    "num_actors": options.num_actors,  # number of training actors
    "gpu_per_actor": options.gpu_per_actor,
    "log_path": options.log_dir,
    "seed": options.seed,  # reproducibility
}

simulator = Simulator(**conf_args)

if options.algorithm == 'fedsgd':
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=options.serv_momentum)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        # opt, milestones=[200, 300, 500], gamma=0.5p
        opt, milestones=[2000, 3000, 5000], gamma=0.5
    )

    assert options.local_round == 1, f"fedsgd requires that only one SGD is taken."

    # runtime parameters
    run_args = {
        "model": model,  # global model
        "client_optimizer": 'SGD',  # server_opt, server optimizer
        "server_optimizer": opt,  # client optimizer
        "loss": "crossentropy",  # loss funcstion
        "global_rounds": options.global_round,  # number of global rounds
        "local_steps": options.local_round,  # number of seps "client_lr": 0.1,  # learning rateteps per round
        "client_lr": 1.0,
        "validate_interval": options.validate_interval,
        "server_lr_scheduler": lr_scheduler,
        "dp_kws": {"clip_threshold": options.clip_threshold, "noise_factor": privacy_factor} if options.dp 
                    else {}
    }


elif options.algorithm == 'fedavg':
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[options.global_round / 3, options.global_round / 2, 2 * options.global_round / 3], gamma=0.5
    )
    server_opt = torch.optim.SGD(model.parameters(), lr=1.0)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     opt, milestones=[200, 300, 500], gamma=0.5
    # )
    # runtime parameters
    run_args = {
        "model": model,  # global model
        "server_optimizer": server_opt,  # server_opt, server optimizer
        "client_optimizer": opt,  # client optimizer
        "loss": "crossentropy",  # loss funcstion
        "global_rounds": options.global_round,  # number of global rounds
        "local_steps": options.local_round,  # number of seps "client_lr": 0.1,  # learning rateteps per round
        # "server_lr": 1.0,
        "validate_interval": 20,
        "client_lr_scheduler": lr_scheduler,
    }

else:
    raise NotImplementedError


simulator.run(**run_args)
