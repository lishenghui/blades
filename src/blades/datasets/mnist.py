from typing import Optional

import numpy as np
import torchvision
from sklearn.utils import shuffle

from .basedataset import BaseDataset


class MNIST(BaseDataset):
    """
    """
    
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
        super(MNIST, self).__init__(data_root, cache_name, train_bs, iid, alpha, num_clients, seed)
    
    def generate_datasets(self, path='./data', iid=True, alpha=0.1, num_clients=20, seed=1):
        train_set = torchvision.datasets.MNIST(train=True, download=True, root=path)
        test_set = torchvision.datasets.MNIST(train=False, download=True, root=path)
        x_test, y_test = test_set.data.numpy(), test_set.targets.numpy()
        x_train, y_train = train_set.data.numpy(), train_set.targets.numpy()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
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