from blades.core import ActorManager
import torch
import torch.nn as nn
from blades.datasets import MNIST
import ray


def test_actormanager():
    lr = 0.1
    net = nn.Sequential(nn.Linear(2, 2))
    opt = torch.optim.SGD(net.parameters(), lr=lr)

    dataset = MNIST()
    actor_mgr = ActorManager.remote(
        net, opt, lr, num_actors=3, gpu_per_actor=0, dataset=dataset
    )

    print(ray.get(actor_mgr.get_shared_updates.remote()))
    ray.get(actor_mgr.update.remote())
    print(ray.get(actor_mgr.get_shared_updates.remote()))


if __name__ == "__main__":
    test_actormanager()
