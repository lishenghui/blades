from pathlib import Path
from torch.nn import Module
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.annotations import PublicAPI
from ray.tune.registry import _global_registry

from fllib.constants import DEFAULT_DATA_ROOT
from fllib.datasets.cifar10 import CIFAR10
from fllib.datasets.fldataset import FLDataset
from fllib.datasets.mnist import MNIST
from fllib.constants import FLLIB_DATASET
from fllib.datasets.fashionmnist import FASHIONMNIST

__all__ = ["CIFAR10", "MNIST", "FASHIONMNIST"]


def make_dataset(
    identifier: str,
    num_clients: int,
    train_batch_size: int = 32,
    seed: int = 0,
    iid: bool = True,
    alpha: float = 0.1,
    **kwargs,
) -> FLDataset:
    data_root: Path = (
        Path.home() / DEFAULT_DATA_ROOT
    )  # typing 'data_root' as a Path object
    valid_datasets = ["cifar10", "mnist", "fashionmnist"]
    if identifier not in valid_datasets:
        raise ValueError(f"Unknown dataset: {identifier}")
    dataset = globals()[identifier.upper()](
        data_root=data_root,
        # cache_name=f"{identifier}.obj",  # simplified cache name generation
        train_bs=train_batch_size,
        num_clients=num_clients,
        seed=seed,
        iid=iid,
        alpha=alpha,
    )  # built-in federated dataset
    return dataset


class DatasetCatalog:
    @staticmethod
    def get_dataset(dataset_config: type = None, **dataset_kwargs) -> Module:
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

            if "num_clients" in dataset_config:
                customized_dataset_kwargs["num_clients"] = dataset_config["num_clients"]
            dataset = dataset_cls(**customized_dataset_kwargs)
        elif dataset_config.get("type") and dataset_config.get("num_clients"):
            dataset = make_dataset(
                dataset_config.get("type"),
                dataset_config.get("num_clients"),
                train_batch_size=dataset_config.get("train_batch_size"),
                seed=dataset_config.get("seed"),
                iid=dataset_config.get("iid", True),
                alpha=dataset_config.get("alpha", 0.1),
            )
        else:
            raise NotImplementedError
        return dataset

    @staticmethod
    @PublicAPI
    def register_custom_dataset(dataset_name: str, dataset_class: type) -> None:
        """Register a custom model class by name.

        The model can be later used by specifying {"custom_model": model_name}
        in the model config.

        Args:
            model_name: Name to register the model under.
            model_class: Python class of the model.
        """
        _global_registry.register(FLLIB_DATASET, dataset_name, dataset_class)
