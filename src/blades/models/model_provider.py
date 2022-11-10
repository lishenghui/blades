from pytorchcv.model_provider import get_model as ptcv_get_model

# import torch
from .resnet_cifar import resnet20


def get_model(name, pretrained=False):
    if "resnet20" == name:
        net = resnet20()
    else:
        net = ptcv_get_model(
            name, pretrained=pretrained, in_size=(32, 32), num_classes=10
        )
    return net
