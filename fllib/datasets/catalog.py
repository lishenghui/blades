from typing import Dict

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.from_config import from_config
from ray.tune.registry import _global_registry

from fllib.constants import FLLIB_DATASET
from fllib.datasets.splitters import IIDSplitter
from fllib.types import DatasetConfigDict
from .dataset import FLDataset
from .ucihar import UCIHAR

# @dataclass
# class DatasetConfig:
#     def __init__(self):
#         splitter_config = {}


_FLLIB_DATASETS = ["UCIHAR"]


CIFAR10_stats = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.2023, 0.1994, 0.2010),
}


torchvision_transforms = {
    "CIFAR10": {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),  # Convert PIL.Image to Tensor
                transforms.Normalize(
                    mean=CIFAR10_stats["mean"], std=CIFAR10_stats["std"]
                ),
                transforms.RandomErasing(p=0.25),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),  # 将PIL.Image转换为张量
                transforms.Normalize(
                    mean=CIFAR10_stats["mean"], std=CIFAR10_stats["std"]
                ),
            ]
        ),
    },
    "MNIST": {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),  # Normalize the data with mean and std deviation
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),  # Convert PIL.Image to Tensor
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),  # Normalize the data with mean and std deviation
            ]
        ),
    },
    "FashionMNIST": {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # Normalizing with mean=0.5 and std=0.5
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # Normalizing with mean=0.5 and std=0.5
            ]
        ),
    },
}


class DatasetCatalog:
    _torch_valid_datasets = ["CIFAR10", "MNIST", "FashionMNIST"]

    @staticmethod
    def from_torch(dataset_config: Dict = None):
        dataset_name = dataset_config.get("type", None)
        if dataset_name not in DatasetCatalog._torch_valid_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        train_set = getattr(datasets, dataset_name)(
            root="~/fldata",
            train=True,
            transform=torchvision_transforms[dataset_name]["train"],
            download=True,
        )
        test_set = getattr(datasets, dataset_name)(
            root="~/fldata",
            train=False,
            transform=torchvision_transforms[dataset_name]["test"],
            download=True,
        )
        splitter_config = dataset_config.pop("splitter_config", {})
        train_batch_size = dataset_config.pop("train_batch_size", 32)
        test_batch_size = dataset_config.pop("test_batch_size", 32)
        splitter = from_config(IIDSplitter, splitter_config)
        subsets = splitter.generate_client_datasets(
            train_set,
            test_set,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
        )
        return FLDataset(subsets)

    @staticmethod
    def _validate_config(config: DatasetConfigDict) -> None:
        pass

    @staticmethod
    def get_dataset(dataset_config: Dict = None, **dataset_kwargs) -> FLDataset:
        DatasetCatalog._validate_config(dataset_config)
        if dataset_config.get("custom_dataset"):
            # Allow model kwargs to be overridden / augmented by
            # custom_model_config.
            customized_dataset_kwargs = dict(
                dataset_kwargs, **dataset_config.get("custom_dataset_config", {})
            )
            if isinstance(dataset_config["custom_dataset"], type):
                dataset_cls = dataset_config["custom_dataset"]
            elif (
                isinstance(dataset_config["custom_dataset"], str)
                and "." in dataset_config["custom_dataset"]
            ):
                return from_config(
                    cls=dataset_config["custom_dataset"],
                    **customized_dataset_kwargs,
                )
            else:
                dataset_cls = _global_registry.get(
                    FLLIB_DATASET, dataset_config["custom_dataset"]
                )

            dataset = dataset_cls(**customized_dataset_kwargs)
            return dataset
        if dataset_config.get("type", None) in _FLLIB_DATASETS:
            if dataset_config.get("type", None) == "UCIHAR":
                dataset = UCIHAR(**dataset_kwargs)
                return dataset
        else:
            return DatasetCatalog.from_torch(dataset_config)

    @staticmethod
    @PublicAPI
    def register_custom_dataset(dataset_name: str, dataset_class: type) -> None:
        """Register a custom model class by name.

        The model can be later used by specifying {"custom_model": model_name}
        in the model config.

        Args:
            dataset_name: Name to register the model under.
            dataset_class: Python class of the dataset.
        """
        _global_registry.register(FLLIB_DATASET, dataset_name, dataset_class)
