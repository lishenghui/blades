# import torch
from ray.rllib.utils.annotations import PublicAPI
from ray.tune.registry import _global_registry
from torch.nn import Module

from fllib.constants import FLLIB_MODEL
from fllib.models.backbones.cctnets import cct_2_3x2_32
from fllib.models.cifar10.resnet_cifar import ResNet10
from fllib.models.fashionmnist.cnn import FashionCNN
from fllib.models.mnist.mlp import MLP


class ModelCatalog:
    @staticmethod
    def get_model(model_config: type = None) -> Module:
        if isinstance(model_config, str):
            if "cct" in model_config:
                model = cct_2_3x2_32()
            elif "resnet" in model_config:
                model = ResNet10()
            elif "mlp" in model_config:
                model = MLP()
            elif "harcnn" in model_config:
                from fllib.models.ucihar.harcnn import HARCNN

                model = HARCNN()
            elif "cnn" in model_config:
                model = FashionCNN()

        elif isinstance(model_config, Module):
            model = model_config
        elif model_config.get("custom_model"):
            model_cls = _global_registry.get(FLLIB_MODEL, model_config["custom_model"])
            model = model_cls()
        else:
            raise NotImplementedError
        return model

    @staticmethod
    @PublicAPI
    def register_custom_model(model_name: str, model_class: type) -> None:
        """Register a custom model class by name.

        The model can be later used by specifying {"custom_model": model_name}
        in the model config.

        Args:
            model_name: Name to register the model under.
            model_class: Python class of the model.
        """
        _global_registry.register(FLLIB_MODEL, model_name, model_class)
