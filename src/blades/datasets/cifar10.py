import os
import pickle
from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.utils import shuffle

from blades.utils import set_random_seed
from .CustomDataset import CustomTensorDataset


class CIFAR10:
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
    
    img_size = 32
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(img_size, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    # ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandAugment(num_ops=1, magnitude=8),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
        transforms.RandomErasing(p=0.25)
    ])
    
    def __init__(
            self,
            data_root: str = './data',
            train_bs: Optional[int] = 32,
            iid: Optional[bool] = True,
            alpha: Optional[float] = 0.1,
            num_clients: Optional[int] = 20
    ):
        self.train_bs = train_bs
        self._data_path = os.path.join(data_root, self.__class__.__name__ + '.obj')
        if not os.path.exists(self._data_path):
            self._generate_datasets(data_root, iid, alpha, num_clients)
    
    def _generate_datasets(self, path='./data', iid=True, alpha=0.1, num_clients=20):
        num_train = 50000
        num_test = 10000
        train_set = torchvision.datasets.CIFAR10(train=True, download=True, root=path)
        test_set = torchvision.datasets.CIFAR10(train=False, download=True, root=path)
        x_test, y_test = test_set.data[:num_test], np.array(test_set.targets)[:num_test]
        x_train, y_train = train_set.data[:num_train], np.array(train_set.targets)[:num_train]
        
        x_train = x_train.astype('float32') / 255.0
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        x_test = x_test.astype('float32') / 255.0
        x_test = np.transpose(x_test, (0, 3, 1, 2))
        
        np.random.seed(1234)
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)
        
        train_user_ids = [str(id) for id in range(num_clients)]
        x_test_splits = np.split(x_test, num_clients)
        y_test_splits = np.split(y_test, num_clients)
        
        if iid:
            x_train_splits = np.split(x_train, num_clients)
            y_train_splits = np.split(y_train, num_clients)
        else:
            print('generating non-iid data')
            min_size = 0
            K = 10
            N = y_train.shape[0]
            client_dataidx_map = {}
            
            while min_size < 10:
                proportion_list = []
                idx_batch = [[] for _ in range(num_clients)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                    
                    proportions = np.array(
                        [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    print(proportions)
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    proportion_list.append(proportions)
            x_train_splits, y_train_splits = [], []
            for j in range(num_clients):
                np.random.shuffle(idx_batch[j])
                client_dataidx_map[j] = idx_batch[j]
                x_train_splits.append(x_train[idx_batch[j], :])
                y_train_splits.append(y_train[idx_batch[j], :])
        
        test_dataset = {}
        train_dataset = {}
        for id, index in zip(train_user_ids, range(num_clients)):
            train_dataset[id] = {'x': x_train_splits[index], 'y': y_train_splits[index].flatten()}
            test_dataset[id] = {'x': x_test_splits[index], 'y': y_test_splits[index].flatten()}
        
        with open(self._data_path, 'wb') as f:
            pickle.dump(train_user_ids, f)
            pickle.dump(train_dataset, f)
            pickle.dump(train_user_ids, f)
            pickle.dump(test_dataset, f)
    
    @staticmethod
    def _preprocess_train_data(
            data,
            labels,
            batch_size,
            seed=0
    ) -> (torch.Tensor, torch.LongTensor):
        # The following line is needed for reproducing the randomness of transforms.
        set_random_seed(seed)
        i = 0
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
                yield CIFAR10.train_transform(X), torch.LongTensor(y)
    
    @staticmethod
    def _preprocess_test_data(
            data,
            labels,
    ) -> CustomTensorDataset:
        tensor_x = torch.Tensor(data)  # transform to torch tensor
        tensor_y = torch.LongTensor(labels)
        return CustomTensorDataset(tensor_x, tensor_y, transform_list=CIFAR10.test_transform)
    
    # generate two lists of dataloaders for train
    def get_dls(self):
        assert os.path.isfile(self._data_path)
        with open(self._data_path, 'rb') as f:
            (train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
        
        assert sorted(train_clients) == sorted(test_clients)
        
        train_dls = []
        test_dls = []
        for idx, u_id in enumerate(train_clients):
            train_dls.append(self._preprocess_train_data(data=np.array(train_data[u_id]['x']),
                                                         labels=np.array(train_data[u_id]['y']),
                                                         batch_size=self.train_bs,
                                                         ))
            test_dls.append(self._preprocess_test_data(data=np.array(test_data[u_id]['x']),
                                                       labels=np.array(test_data[u_id]['y']),
                                                       ))
        return train_dls, test_dls
