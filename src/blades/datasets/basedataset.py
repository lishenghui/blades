import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from blades.utils.utils import set_random_seed
from .customdataset import CustomTensorDataset

logger = logging.getLogger(__name__)

class BaseDataset(ABC):
    train_transform = None
    test_transform = None
    def __init__(
            self,
            data_root: str = './data',
            cache_name: str = "",
            train_bs: Optional[int] = 32,
            iid: Optional[bool] = True,
            alpha: Optional[float] = 0.1,
            num_clients: Optional[int] = 20,
            seed = 1,
    ):
        self.train_bs = train_bs
        if cache_name == "":
            cache_name = self.__class__.__name__
        self._data_path = os.path.join(data_root, cache_name + '.obj')
        
        # Meta parameters for data partitioning, for comparison with cache. Regenerate dataset
        # if those parameters are different.
        meta_info = {
            "num_clients": num_clients,
            "data_root": data_root,
            "train_bs": train_bs,
            "iid": iid,
            "alpha": alpha,
            "seed": seed,
        }
        
        regenerate = True
        if os.path.exists(self._data_path):
            with open(self._data_path, 'rb') as f:
                loaded_meta_info  = pickle.load(f)
                if loaded_meta_info == meta_info:
                    regenerate = False
                else:
                    logger.warning(
                        "arguments for data partitioning didn't match the cache,"
                        " datasets will be regenerated using the new setting."
                    )

        if regenerate:
            returns = self.generate_datasets(data_root, iid, alpha, num_clients, seed)
            with open(self._data_path, 'wb') as f:
                pickle.dump(meta_info, f)
                for obj in returns:
                    pickle.dump(obj, f)
                    
    
    @abstractmethod
    def generate_datasets(self, path='./data', iid=True, alpha=0.1, num_clients=20, seed=1):
        pass
    
    def _preprocess_train_data(
            self,
            data,
            labels,
            batch_size,
            seed=0
    ) -> (torch.Tensor, torch.LongTensor):
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
                
                continue
            else:
                X = data[i * batch_size:(i + 1) * batch_size, :]
                y = labels[i * batch_size:(i + 1) * batch_size]
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
        return CustomTensorDataset(tensor_x, tensor_y, transform_list=self.train_transform)
    
    # generate two lists of dataloaders for train
    def get_dls(self):
        assert os.path.isfile(self._data_path)
        with open(self._data_path, 'rb') as f:
            (_, train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(5)]
        
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

