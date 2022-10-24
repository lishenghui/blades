import os

import ray
import torch

from blades.aggregators import init_aggregator
from blades.attackers import init_attacker
from args import options
from simulator import Simulator
from blades.datasets import CIFAR10
from blades.models import CCTNet10
from blades.clients import BladesClient
from blades.servers import BladesServer

args = options
if not ray.is_initialized():
    ray.init()
else:
    ray.init(address="auto")

if not os.path.exists(options.log_dir):
    os.makedirs(options.log_dir)

# Use absolute path name, as Ray might have trouble in fetching relative path
data_root = os.path.abspath("./data")
cache_name = (
    options.dataset
    + "_"
    + options.algorithm
    + ("_noniid" if not options.non_iid else "")
    + f"_{str(options.num_clients)}_{str(options.seed)}"
    + ".obj"
)
if options.dataset == "cifar10":
    dataset = CIFAR10(
        data_root=data_root,
        cache_name=cache_name,
        train_bs=options.batch_size,
        num_clients=options.num_clients,
        iid=not options.non_iid,
        seed=0,
    )  # built-in federated cifar10 dataset
    model = CCTNet10()

else:
    raise NotImplementedError

if options.gpu_per_actor > 0.0:
    model = model.to("cuda")

num_clients = options.num_clients
num_byzantine = options.num_byzantine
device = "cuda"
net = CCTNet10().to(device)
local_opt_cls = torch.optim.SGD
local_opt_kws = {"lr": 1.0, "momentum": 0, "dampening": 0}
dataset = CIFAR10(train_bs=64, num_clients=num_clients, seed=0)

clients = [BladesClient(id=str(id)) for id in range(num_clients)]

for i in range(num_byzantine):

    clients[i] = init_attacker(options.attack, {"id": str(i)} | options.attack_kws)
agg = init_aggregator(options.agg, options.aggregator_kws)
world_size = 0

server_kws = {
    "model": net,
    "opt_cls": torch.optim.SGD,
    "opt_kws": {"lr": 0.1, "momentum": 0.9, "dampening": 0},
    "aggregator": agg,
}
runner = Simulator(
    dataset=dataset,
    clients=clients,
    num_gpus=2,
    num_gpus_mgr=0.2,
    num_actors_mgr=5,
    num_gpus_actor=0.15,
    local_opt_cls=local_opt_cls,
    local_opt_kws=local_opt_kws,
    global_model=net,
    server_cls=BladesServer,
    server_kws=server_kws,
    log_path=options.log_dir,
)

runner.run()
