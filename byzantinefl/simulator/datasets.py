import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Any, Callable, Optional, Tuple


def preprocess_data(data, labels, batch_size, augmentations, seed=0):
    i = 0
    np.random.seed(seed=seed)
    idx = np.random.permutation(len(labels))
    data, labels = data[idx], labels[idx]
    
    while True:
        if i * batch_size >= len(labels):
            i = 0
            idx = np.random.permutation(len(labels))
            data, labels = data[idx], labels[idx]
            
            continue
        else:
            X = data[i * batch_size:(i + 1) * batch_size, :]
            y = labels[i * batch_size:(i + 1) * batch_size]
            i += 1
            X = torch.Tensor(X)
            yield augmentations(X), torch.LongTensor(y)
      


    
    
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


class DatasetBase(object):
    def __init__(
        self,
        data_path: str,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        train_bs=32
    ) -> None:
        self.train_sets = {}
        self.test_sets = {}
        assert os.path.isfile(data_path)
        with open(data_path, 'rb') as f:
            (train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
        
        assert sorted(train_clients) == sorted(test_clients)
        self.clients = train_clients
        for idx, u_id in enumerate(train_clients):
            self.train_sets[u_id] = self._preprocess_train_data(data = np.array(train_data[u_id]['x']),
                                                                labels = np.array(train_data[u_id]['y']),
                                                                batch_size=train_bs,
                                                                transform=train_transform
                                                                )
            self.test_sets[u_id] = self._preprocess_test_data(data = np.array(test_data[u_id]['x']),
                                                              labels = np.array(test_data[u_id]['y']),
                                                              transform=test_transform
                                                              )
    
    def _preprocess_train_data(
        self,
        data,
        labels,
        batch_size,
        transform: Optional[Callable] = None,
        seed=0
    ) -> (torch.Tensor, torch.LongTensor):
        i = 0
        np.random.seed(seed=seed)
        idx = np.random.permutation(len(labels))
        data, labels = data[idx], labels[idx]
    
        if transform:
            augmentations = transform
        else:
            augmentations = transforms.Compose([])
        
        while True:
            if i * batch_size >= len(labels):
                i = 0
                idx = np.random.permutation(len(labels))
                data, labels = data[idx], labels[idx]
            
                continue
            else:
                X = data[i * batch_size:(i + 1) * batch_size, :]
                y = labels[i * batch_size:(i + 1) * batch_size]
                i += 1
                X = torch.Tensor(X)
                yield augmentations(X), torch.LongTensor(y)

    def _preprocess_test_data(
            self,
            data,
            labels,
            transform: Optional[Callable] = None,
    ) -> Dataset:
        if transform:
            augmentations = transform
        else:
            augmentations = transforms.Compose([])
    
        mean = (0.4914, 0.4822, 0.4465),
        std = (0.2023, 0.1994, 0.2010),
        test_transform0 = transforms.Compose([
            transforms.Normalize(mean, std),
        ])
    
        cifar10_stats = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        }
        test_transform = transforms.Compose([
            transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
            # transforms.Normalize(mean, std),
        ])
    
        tensor_x = torch.Tensor(data)  # transform to torch tensor
        tensor_y = torch.LongTensor(labels)
        return CustomTensorDataset(tensor_x, tensor_y, transform_list=test_transform)
    
    def get_train_data(self, u_id, num_batchs):
        data = [next(self.train_sets[u_id]) for _ in range(num_batchs)]
        return data


        
    def get_all_test_data(self, u_id):
        return self.test_sets[u_id]


class CIFAR10(DatasetBase):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

        Args:
            path (string): Root directory of dataset where directory
                ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.

    """
    stats = {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    }
    test_transform = transforms.Compose([
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])
    
    def __init__(self,
        data_path: str,
        train_bs=32
        ):
        super().__init__(
            data_path=data_path,
            train_transform=self.train_transform,
            test_transform=self.test_transform,
            train_bs=train_bs
        )


if __name__ == "__main__":
    dataset = CIFAR10('../tasks/cifar10/data/data_cache.obj')