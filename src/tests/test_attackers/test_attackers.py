import copy

import torch
from torch import nn

from blades.attackers.signflippingclient import SignflippingClient
from blades.core import BladesClient
from blades.utils.utils import (
    set_random_seed,
)


def test_signflipping():
    set_random_seed(0)
    net = nn.Sequential(nn.Linear(2, 2))
    state_dic = copy.deepcopy(net.state_dict())
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    data = torch.rand(2, 2)
    target = torch.randint(0, 1, (2,))
    dataset_gen = (i for i in [(data, target)])

    malicious_client = SignflippingClient()
    malicious_client.set_loss()
    malicious_client.set_global_model_ref(net)
    malicious_client.train_global_model(dataset_gen, 1, opt)

    dataset_gen = (i for i in [(data, target)])
    net.load_state_dict(state_dic)
    benign_client = BladesClient()
    benign_client.set_loss()
    benign_client.set_global_model_ref(net)
    benign_client.train_global_model(dataset_gen, 1, opt)

    assert torch.equal(benign_client.get_update(), -malicious_client.get_update())


if __name__ == "__main__":
    test_signflipping()
