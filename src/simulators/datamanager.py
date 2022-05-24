import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset

from settings.data_utils import preprocess_data


class DataManager(object):
    def __init__(self, data_path, train_bs, test_bs, *args, **kwargs):
        self.train_sets = {}
        self.test_sets = {}
        assert os.path.isfile(data_path)
        with open(data_path, 'rb') as f:
            (train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
        
        assert sorted(train_clients) == sorted(test_clients)
        self.clients = train_clients
        for idx, u_id in enumerate(train_clients):
            self.train_sets[u_id] = preprocess_data(np.array(train_data[u_id]['x']), np.array(train_data[u_id]['y']),
                                                    batch_size=train_bs)
            self.test_sets[u_id] = self.__build_testset(test_data[u_id])
    
    def __build_testset(self, data):
        tensor_x = torch.Tensor(data['x'])  # transform to torch tensor
        tensor_y = torch.LongTensor(data['y'])
        
        dataset = TensorDataset(tensor_x, tensor_y)  # create your dataset
        return dataset
    
    def get_train_data(self, u_id, num_batchs):
        data = [next(self.train_sets[u_id]) for _ in range(num_batchs)]
        return data
    
    def get_all_test_data(self, u_id):
        return self.test_sets[u_id]
