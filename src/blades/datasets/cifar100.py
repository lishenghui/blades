from typing import Optional

import numpy as np
import torchvision
import torchvision.transforms as transforms

from .fldataset import FLDataset


class CIFAR100(FLDataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        path (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to
            True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        data_root: str = "./data",
        cache_name: str = "",
        iid: Optional[bool] = True,
        alpha: Optional[float] = 0.1,
        num_clients: Optional[int] = 20,
        seed=1,
        train_data=None,
        test_data=None,
        train_bs: Optional[int] = 32,
    ):
        train_set = torchvision.datasets.CIFAR100(
            train=True, download=True, root=data_root
        )
        test_set = torchvision.datasets.CIFAR100(
            train=False,
            download=True,
            root=data_root,
        )
        stats = {
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
        }
        x_test, y_test = test_set.data, np.array(test_set.targets)
        x_train, y_train = train_set.data, np.array(train_set.targets)

        x_train = np.transpose(x_train, (0, 3, 1, 2))
        x_test = np.transpose(x_test, (0, 3, 1, 2))

        test_transform = transforms.Compose(
            [
                transforms.Normalize(mean=stats["mean"], std=stats["std"]),
            ]
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize(mean=stats["mean"], std=stats["std"]),
                transforms.RandomErasing(p=0.25),
            ]
        )

        super(CIFAR100, self).__init__(
            (x_train, y_train),
            (x_test, y_test),
            data_root=data_root,
            cache_name=cache_name,
            iid=iid,
            num_classes=100,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
            train_data=train_data,
            test_data=test_data,
            train_bs=train_bs,
            train_transform=train_transform,
            test_transform=test_transform,
        )
