import copy

import torch
import torch.nn as nn

from blades.clients import RSAClient
from blades.core import BladesClient


def test_rsaclient():
    lr = 0.1
    lambda_ = 0.1
    net = nn.Sequential(nn.Linear(2, 2))
    opt = torch.optim.SGD(net.parameters(), lr=lr)

    per_net = copy.deepcopy(net)
    per_opt = torch.optim.SGD(per_net.parameters(), lr=lr)

    rsa_client = RSAClient(per_net, per_opt, lambda_)
    rsa_client.set_loss()
    data = torch.rand(2, 2)
    target = torch.randint(0, 1, (2,))
    dataset_gen = (i for i in [(data, target)])

    rsa_client.train_personal_model(dataset_gen, 1, net.state_dict())

    dataset_gen = (i for i in [(data, target)])
    base_client = BladesClient()
    base_client.set_loss()
    base_client.set_global_model_ref(net)
    base_client.train_global_model(dataset_gen, 1, opt)

    for (_, p_rsa), (_, p_base) in zip(
        rsa_client._per_model.named_parameters(), net.named_parameters()
    ):
        assert torch.allclose(p_rsa, p_base)
