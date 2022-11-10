from .cifar10.cct import CCTNet10
from .cifar100.cct import CCTNet100
from .mnist.mlp import MLP

from .model_provider import get_model

__all__ = ["CCTNet10", "CCTNet100", "MLP", "get_model"]
