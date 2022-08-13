from typing import Optional

import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.utils import shuffle

from .basedataset import BaseDataset


class CIFAR10(BaseDataset):
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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
        transforms.RandomErasing(p=0.25)
    ])
    
    def __init__(
            self,
            data_root: str = './data',
            cache_name: str = "",
            train_bs: Optional[int] = 32,
            iid: Optional[bool] = True,
            alpha: Optional[float] = 0.1,
            num_clients: Optional[int] = 20,
            seed: Optional[int] = 1,
    ):
        super(CIFAR10, self).__init__(data_root, cache_name, train_bs, iid, alpha, num_clients, seed)
    
    def generate_datasets(self, path='./data', iid=True, alpha=0.1, num_clients=20, seed=1):
        train_set = torchvision.datasets.CIFAR10(train=True, download=True, root=path)
        test_set = torchvision.datasets.CIFAR10(train=False, download=True, root=path)
        x_test, y_test = test_set.data, np.array(test_set.targets)
        x_train, y_train = train_set.data, np.array(train_set.targets)
        
        x_train = x_train.astype('float32') / 255.0
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        x_test = x_test.astype('float32') / 255.0
        x_test = np.transpose(x_test, (0, 3, 1, 2))
        
        np.random.seed(seed)
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
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
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
            train_dataset[id] = {'x': x_train_splits[index], 'y': y_train_splits[index].flatten()}
            test_dataset[id] = {'x': x_test_splits[index], 'y': y_test_splits[index].flatten()}
        
        return train_user_ids, train_dataset, train_user_ids, test_dataset