from typing import Optional

import torchvision

from .fldataset import FLDataset


class MNIST(FLDataset):
    """"""

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
        train_set = torchvision.datasets.MNIST(
            train=True, download=True, root=data_root
        )
        test_set = torchvision.datasets.MNIST(
            train=False, download=True, root=data_root
        )
        x_test, y_test = test_set.data.numpy(), test_set.targets.numpy()
        x_train, y_train = train_set.data.numpy(), train_set.targets.numpy()
        super(MNIST, self).__init__(
            (x_train, y_train),
            (x_test, y_test),
            data_root=data_root,
            cache_name=cache_name,
            iid=iid,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
            train_data=train_data,
            test_data=test_data,
            train_bs=train_bs,
        )
