from blades.core import ActorManager, Actor
import torch
from blades.datasets import MNIST, CIFAR10
import ray
from blades.servers import BladesServer
from blades.aggregators import Mean
from blades.models import MLP, CCTNet10
from blades.clients import BladesClient
import logging
import sys
from tqdm import trange
import numpy as np
import os
from blades.utils.utils import (
    # initialize_logger,
    # reset_model_weights,
    # set_random_seed,
    top1_accuracy,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test_actor():
    net = MLP()

    dataset = MNIST()
    # dataset = CIFAR10()
    opt_cls = torch.optim.SGD
    opt_kws = {"lr": 0.1, "momentum": 0, "dampening": 0}
    client = BladesClient(id="0", device="cuda")
    actor = Actor.remote(
        dataset,
        net,
        opt_cls,
        opt_kws,
    )

    clients = ray.get(actor.local_train.remote(clients=[client] * 2, global_model=net))
    updates = [client.get_update() for client in list(clients)]
    assert torch.allclose(updates[0], updates[1])


def log_validate(metrics):
    top1 = np.average(
        [metric["top1"] for metric in metrics],
        weights=[metric["Length"] for metric in metrics],
    )
    loss = np.average(
        [metric["Loss"] for metric in metrics],
        weights=[metric["Length"] for metric in metrics],
    )
    return loss, top1


def test_actormanager():
    logger.info("starting ...")

    device = "cuda"
    net = CCTNet10().to(device)
    opt_cls = torch.optim.SGD
    opt_kws = {"lr": 1.0, "momentum": 0, "dampening": 0}
    clients = [BladesClient(id=str(id)) for id in range(50)]
    dataset = CIFAR10(num_clients=50)
    agg = Mean()
    world_size = 0

    server_kws = {
        "model": net,
        "opt_cls": opt_cls,
        "opt_kws": {"lr": 0.1, "momentum": 0.9, "dampening": 0},
        "aggregator": agg,
    }
    actor_mgr = ActorManager.options(num_gpus=0.2).remote(
        dataset,
        net,
        opt_cls,
        opt_kws,
        num_actors=8,
        num_buffers=len(clients),
        gpu_per_actor=0.07,
        world_size=world_size,
        server_cls=BladesServer,
        server_kws=server_kws,
        device=device,
    )

    global_rounds = 1000
    validate_interval = 100
    with trange(0, global_rounds + 1) as t:
        for global_rounds in t:
            ret_actor_mgr = actor_mgr.train.remote(clients=clients)
            ray.get([ret_actor_mgr])

            if global_rounds % validate_interval == 0 and global_rounds > 0:
                ret_actor_mgr = actor_mgr.evaluate.remote(
                    clients=clients,
                    round_number=global_rounds,
                    metrics={"top1": top1_accuracy},
                )
                ret_test = ray.get(ret_actor_mgr)
                test_results = log_validate(ret_test)
                t.set_postfix(loss=test_results[0], top1=test_results[1])


def test_actormanager_cross_GPU():
    logger.info("starting ...")

    device = "cuda"
    net = CCTNet10().to(device)
    opt_cls = torch.optim.SGD
    opt_kws = {"lr": 1.0, "momentum": 0, "dampening": 0}
    clients = [BladesClient(id=str(id)) for id in range(50)]
    dataset = CIFAR10(num_clients=50)
    agg = Mean()

    server_kws = {
        "model": net,
        "opt_cls": opt_cls,
        "opt_kws": {"lr": 0.1, "momentum": 0.9, "dampening": 0},
        "aggregator": agg,
    }

    act_mgrs = []
    num_gpus = 3
    for i in range(num_gpus):
        if i == 0:
            server_cls = BladesServer
            num_gpus_mgr = 0.2
        else:
            server_cls = None
            num_gpus_mgr = 0.2
        actor_mgr = ActorManager.options(num_gpus=num_gpus_mgr).remote(
            dataset,
            net,
            opt_cls,
            opt_kws,
            rank=i,
            num_actors=2,
            num_buffers=len(clients),
            num_selected_clients=len(clients),
            gpu_per_actor=0.39,
            world_size=num_gpus,
            server_cls=server_cls,
            server_kws=server_kws,
            device=device,
            visible_gpu=str(i),
        )
        ray.get(actor_mgr.init.remote())
        act_mgrs.append(actor_mgr)

    ray.get([mgr.init_dist.remote() for mgr in act_mgrs])
    global_rounds = 40
    validate_interval = 10
    with trange(0, global_rounds + 1) as t:
        for global_rounds in t:
            client_groups = np.array_split(clients, num_gpus)
            ret_ids = [
                actor_mgr.train.remote(clients=client_group)
                for (actor_mgr, client_group) in zip(act_mgrs, client_groups)
            ]
            ray.get(ret_ids)

            if global_rounds % validate_interval == 0 and global_rounds > 0:
                ret_actor_mgr = actor_mgr.evaluate.remote(
                    clients=clients,
                    round_number=global_rounds,
                    metrics={"top1": top1_accuracy},
                )
                ret_test = ray.get(ret_actor_mgr)
                test_results = log_validate(ret_test)
                t.set_postfix(loss=test_results[0], top1=test_results[1])


test_actormanager_cross_GPU()
# test_actormanager()
