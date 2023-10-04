import os
from typing import Optional

import numpy as np
from urllib.error import URLError
from fllib.datasets import FLDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    check_integrity,
)


class UCIHAR(FLDataset):
    """"""

    mirrors = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/",
    ]

    resources = [
        ("UCI%20HAR%20Dataset.zip", None),
    ]

    def __init__(
        self,
        data_root: str = "./data",
        cache_name: str = "",
        iid: Optional[bool] = True,
        alpha: Optional[float] = 0.1,
        num_clients: Optional[int] = 20,
        download: bool = True,
        seed=1,
        train_data=None,
        test_data=None,
        train_bs: Optional[int] = 64,
    ):
        if isinstance(data_root, str):
            data_root = os.path.expanduser(data_root)
        self.data_root = data_root
        if download:
            self.download()
        super(UCIHAR, self).__init__(
            (None, None),
            (None, None),
            data_root=data_root,
            cache_name=cache_name,
            iid=iid,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
            train_data=train_data,
            test_data=test_data,
            train_bs=train_bs,
            is_image=False,
        )

    def _check_exists(self) -> bool:
        return all(
            check_integrity(
                os.path.join(
                    self.raw_folder, os.path.splitext(os.path.basename(url))[0]
                )
            )
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.data_root, self.__class__.__name__, "raw")

    def _load_data(self):
        str_folder = self.raw_folder + "/UCI HAR Dataset/"
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_",
        ]

        str_train_files = [
            str_folder + "train/" + "Inertial Signals/" + item + "train.txt"
            for item in INPUT_SIGNAL_TYPES
        ]
        str_test_files = [
            str_folder + "test/" + "Inertial Signals/" + item + "test.txt"
            for item in INPUT_SIGNAL_TYPES
        ]
        str_train_y = str_folder + "train/y_train.txt"
        str_test_y = str_folder + "test/y_test.txt"
        str_train_id = str_folder + "train/subject_train.txt"
        str_test_id = str_folder + "test/subject_test.txt"

        def format_data_x(datafile):
            x_data = None
            for item in datafile:
                item_data = np.loadtxt(item, dtype=np.float32)
                if x_data is None:
                    x_data = np.zeros((len(item_data), 1))
                x_data = np.hstack((x_data, item_data))
            x_data = x_data[:, 1:]
            X = None
            for i in range(len(x_data)):
                row = np.asarray(x_data[i, :])
                row = row.reshape(9, 128).T
                if X is None:
                    X = np.zeros((len(x_data), 128, 9))
                X[i] = row
            return X

        def format_data_y(datafile):
            return np.loadtxt(datafile, dtype=np.int32) - 1

        def read_ids(datafile):
            return np.loadtxt(datafile, dtype=np.int32)

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)
        id_train = read_ids(str_train_id)
        id_test = read_ids(str_test_id)

        X_train, X_test = X_train.reshape((-1, 9, 1, 128)), X_test.reshape(
            (-1, 9, 1, 128)
        )
        return (X_train, Y_train), (X_test, Y_test), (id_train, id_test)

    def _generate_datasets(
        self, train_set=None, test_set=None, iid=True, alpha=0.1, num_clients=20, seed=1
    ):
        (X_train, Y_train), (X_test, Y_test), (id_train, id_test) = self._load_data()
        id_train_unique = np.unique(id_train)
        id_test_unique = np.unique(id_test)

        train_dataset = {}
        for uid in id_train_unique:
            train_indices = np.where(id_train == uid)[0]
            train_dataset[uid] = {
                "x": X_train[train_indices],
                "y": Y_train[train_indices].flatten(),
            }
        test_dataset = {}
        for uid in id_test_unique:
            test_indices = np.where(id_test == uid)[0]
            if len(test_indices) > 0:
                test_dataset[uid] = {
                    "x": X_test[test_indices],
                    "y": Y_test[test_indices].flatten(),
                }
        return id_train_unique, train_dataset, id_test_unique, test_dataset


if "__main__" == __name__:
    data_root = "./data"
    ucihar = UCIHAR(data_root=data_root, download=True)
    result = ucihar._generate_datasets()
    breakpoint()
    print("done")
