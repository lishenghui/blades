import copy
import logging
import os
import pickle
import random
from abc import ABC
from functools import partial
from pathlib import Path
from typing import List, Optional, Generator, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from fllib.constants import DEFAULT_DATA_ROOT

logger = logging.getLogger(__name__)


def set_random_seed(seed_value=0, use_cuda=False):
    np.random.seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CustomTensorDataset(Dataset):
    def __init__(self, data_X, data_y, transform_list=None):
        tensors = (data_X, data_y)
        self.tensors = tensors
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transforms:
            x = self.transforms(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[1].size(0)


class FLDataset(ABC):
    def __init__(
        self,
        train_set=None,
        test_set=None,
        data_root: str = "",
        cache_name: str = "",
        iid: Optional[bool] = True,
        alpha: Optional[float] = 0.1,
        num_clients: Optional[int] = 20,
        num_classes: Optional[int] = 10,
        seed=1,
        train_data=None,
        test_data=None,
        train_bs: Optional[int] = 32,
        test_bs: Optional[int] = 128,
        train_transform=None,
        test_transform=None,
        is_image=True,
    ):
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_clients = num_clients
        if train_data:
            self.train_data = train_data
            self.test_data = test_data
            self.test_bs = test_bs
            self.train_bs = train_bs
            self._preprocess()
            return
        self.num_classes = num_classes
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.is_image = is_image
        if cache_name == "":
            cache_name = (
                f"{self.__class__.__name__}_num_clients_{num_clients}_iid_{iid}"
                f"_alpha_{alpha}_train_bs_{train_bs}_seed_{seed}.obj"
            )
        if data_root == "":
            data_root = (
                Path.home() / DEFAULT_DATA_ROOT
            )  # typing 'data_root' as a Path object
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        self._data_path = os.path.join(data_root, cache_name)

        # Meta parameters for data partitioning, for comparison with cache.
        # Regenerate dataset if those parameters are different.
        meta_info = {
            "num_clients": num_clients,
            "data_root": data_root,
            "train_bs": train_bs,
            "iid": iid,
            "alpha": alpha,
            "seed": seed,
            "is_image": is_image,
        }

        regenerate = True
        if os.path.exists(self._data_path):
            with open(self._data_path, "rb") as f:
                loaded_meta_info = pickle.load(f)
                if loaded_meta_info == meta_info:
                    regenerate = False
                else:
                    logger.warning(
                        "arguments for data partitioning didn't match the cache,"
                        " datasets will be regenerated using the new setting."
                    )

        if regenerate:
            returns = self._generate_datasets(
                train_set,
                test_set,
                iid=iid,
                alpha=alpha,
                num_clients=num_clients,
                seed=seed,
            )
            with open(self._data_path, "wb") as f:
                pickle.dump(meta_info, f)
                for obj in returns:
                    pickle.dump(obj, f)

        assert os.path.isfile(self._data_path)
        with open(self._data_path, "rb") as f:
            (_, train_clients, self.train_data, test_clients, self.test_data) = [
                pickle.load(f) for _ in range(5)
            ]

        # assert sorted(train_clients) == sorted(test_clients)
        self._preprocess()

    def __reduce__(self):
        deserializer = FLDataset
        return (
            partial(
                deserializer,
                train_data=self.train_data,
                test_data=self.test_data,
                train_bs=self.train_bs,
                num_clients=self.num_clients,
                train_transform=self.train_transform,
                test_transform=self.test_transform,
            ),
            (),
        )

    def _generate_datasets(
        self, train_set, test_set, *, iid=True, alpha=0.1, num_clients=20, seed=1
    ):
        x_test, y_test = test_set
        x_train, y_train = train_set
        if self.is_image:
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0

        np.random.seed(seed)
        # x_train, y_train = shuffle(x_train, y_train)
        # x_test, y_test = shuffle(x_test, y_test)

        train_user_ids = [str(id) for id in range(num_clients)]
        x_test_splits = np.array_split(x_test, num_clients)
        y_test_splits = np.array_split(y_test, num_clients)

        if iid:
            x_train_splits = np.array_split(x_train, num_clients)
            y_train_splits = np.array_split(y_train, num_clients)
        else:
            print("generating non-iid data")
            min_size = 0
            N = y_train.shape[0]
            client_dataidx_map = {}

            while min_size < 10:
                proportion_list = []
                idx_batch = [[] for _ in range(num_clients)]
                for k in range(self.num_classes):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

                    proportions = np.array(
                        [
                            p * (len(idx_j) < N / num_clients)
                            for p, idx_j in zip(proportions, idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                    ]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    proportion_list.append(proportions)
            x_train_splits, y_train_splits = [], []
            for j in range(num_clients):
                np.random.shuffle(idx_batch[j])
                client_dataidx_map[j] = idx_batch[j]
                x_train_splits.append(x_train[idx_batch[j], :])
                y_train_splits.append(y_train[idx_batch[j]])

        test_dataset = {}
        train_dataset = {}
        for id, index in zip(train_user_ids, range(num_clients)):
            train_dataset[id] = {
                "x": x_train_splits[index],
                "y": y_train_splits[index].flatten(),
            }
            test_dataset[id] = {
                "x": x_test_splits[index],
                "y": y_test_splits[index].flatten(),
            }

        return train_user_ids, train_dataset, train_user_ids, test_dataset

    def _preprocess_train_data(
        self, data, labels, batch_size, seed=0
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        i = 0
        # The following line is needed for reproducing the randomness of transforms.
        set_random_seed(seed)

        idx = np.random.permutation(len(labels))
        data, labels = data[idx], labels[idx]

        while True:
            if i * batch_size >= len(labels):
                i = 0
                idx = np.random.permutation(len(labels))
                data, labels = data[idx], labels[idx]
            X = data[i * batch_size : (i + 1) * batch_size, :]
            y = labels[i * batch_size : (i + 1) * batch_size]
            i += 1
            X = torch.Tensor(X)
            if self.train_transform:
                X = self.train_transform(X)
            yield X, torch.LongTensor(y)

    def _preprocess_test_data(
        self,
        data,
        labels,
    ) -> CustomTensorDataset:
        tensor_x = torch.Tensor(data)  # transform to torch tensor
        tensor_y = torch.LongTensor(labels)
        return CustomTensorDataset(
            tensor_x, tensor_y, transform_list=self.test_transform
        )

    def _preprocess(self):
        self._train_dls = {}
        self._test_dls = {}
        for idx, u_id in enumerate(self.train_data.keys()):
            self._train_dls[u_id] = self._preprocess_train_data(
                data=np.array(self.train_data[u_id]["x"]),
                labels=np.array(self.train_data[u_id]["y"]),
                batch_size=self.train_bs,
            )

        for idx, u_id in enumerate(self.test_data.keys()):
            self._test_dls[u_id] = self._preprocess_test_data(
                data=np.array(self.test_data[u_id]["x"]),
                labels=np.array(self.test_data[u_id]["y"]),
            )

    @property
    def client_ids(self):
        return list(set(self.train_client_ids) | set(self.test_client_ids))

    @property
    def train_client_ids(self):
        return list(self._train_dls.keys())

    @property
    def test_client_ids(self):
        return list(self._test_dls.keys())

    def subset(self, u_ids: List[str]):
        subset = copy.deepcopy(self)
        subset._train_dls = {k: v for k, v in self._train_dls.items() if k in u_ids}
        subset._test_dls = {k: v for k, v in self._test_dls.items() if k in u_ids}
        subset.train_data = {k: v for k, v in self.train_data.items() if k in u_ids}
        subset.test_data = {k: v for k, v in self.test_data.items() if k in u_ids}
        return subset

    def split(self, n: int):
        """Randomly split the dataset into `n` disjoint subsets."""

        if n <= 0:
            raise ValueError(f"The number of splits {n} is not positive.")

        keys = list(self._train_dls.keys())
        random.shuffle(keys)

        # Determine the size of each subset
        subset_size = len(keys) // n

        # Divide `keys` into `n` disjoint subsets
        subset_keys = [keys[i * subset_size : (i + 1) * subset_size] for i in range(n)]

        for i in range(len(keys) % n):
            subset_keys[i].append(keys[-(i + 1)])

        subsets = []
        for keys in subset_keys:
            subset = self.subset(keys)
            subsets.append(subset)
        return subsets

    def get_train_loader(self, u_id: str) -> Generator:
        """
        Get the local dataset of given user `id`.
        Args:
            u_id (str): user id.

        Returns: the `generator` of dataset for the given `u_id`.
        """
        # breakpoint()
        return self._train_dls[u_id]

    def get_test_loader(self, u_id):
        return DataLoader(dataset=self._test_dls[u_id], batch_size=self.test_bs)
