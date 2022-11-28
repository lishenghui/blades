import os
import time

import ray
import torch

import wandb
from args import options
from blades.aggregators import get_aggregator
from blades.attackers import init_attacker
from blades.clients import BladesClient
from blades.core import BladesServer
from blades.core.config import ScalingConfig, RunConfig
from blades.core.simulator import Simulator
from blades.datasets.data_provider import get_dataset
from blades.utils.utils import set_random_seed

wandb.init(project="blades", entity="lishenghui")

set_random_seed(0, use_cuda=True)
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

device = torch.device("cuda" if options.gpu_per_actor > 0 else "cpu")

local_opt_cls = torch.optim.SGD
local_opt_kws = {"lr": 1.0, "momentum": 0, "dampening": 0}
dataset = get_dataset(
    options.dataset,
    train_bs=options.batch_size,
    num_clients=options.num_clients,
    seed=0,
)

clients = [
    BladesClient(id=str(id), momentum=options.client_momentum, device=device)
    for id in range(options.num_clients)
]

for i in range(options.num_byzantine):
    clients[i] = init_attacker(
        options.attack,
        {"id": str(i), "momentum": options.client_momentum} | options.attack_kws,
    )

agg = get_aggregator(options.agg, options.aggregator_kws, bucketing=options.bucketing)
world_size = 0


run_config = RunConfig(
    server_cls=BladesServer,
    server_kws={
        "opt_cls": torch.optim.SGD,
        "opt_kws": {"lr": 0.05, "momentum": 0.9, "dampening": 0},
        "aggregator": agg,
    },
    clients=clients,
    local_opt_cls=local_opt_cls,
    local_opt_kws=local_opt_kws,
    global_model=options.model,
)


scaling_config = ScalingConfig(
    num_workers=options.num_actors,
    resources_per_worker={
        "GPU": options.gpu_per_actor,
        "CPU": 1,
    },
    server_resources={
        "GPU": options.num_gpus_server,
        "CPU": 1,
    },
)

t_s = time.time()
runner = Simulator(
    dataset=dataset,
    scaling_config=scaling_config,
    run_config=run_config,
)

print(f"init time: {time.time() - t_s}")

runner.run(
    validate_interval=options.validate_interval,
    global_rounds=options.global_round,
    local_steps=options.local_steps,
)
