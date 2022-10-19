# import torch
import torch.nn as nn
from blades.utils.torch_utils import get_num_params


def test_get_num_params():
    net = nn.Sequential(nn.Linear(2, 2))
    n = get_num_params(net)
    assert n == 6
