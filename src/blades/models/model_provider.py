from pytorchcv.model_provider import get_model as ptcv_get_model

from .cifar10.cct import CCTNet10
from .mnist.mlp import MLP

# import torch
from .resnet_cifar import resnet20


def get_model(name, pretrained=False):
    if name == "resnet20":
        net = resnet20()
    elif name == "CCTNet10":
        net = CCTNet10()
    elif name == "toy_mlp":
        net = MLP()
    else:
        net = ptcv_get_model(
            name, pretrained=pretrained, in_size=(32, 32), num_classes=10
        )
    return net
