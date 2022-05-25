import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from settings.data_utils import preprocess_data
import torchvision.transforms as transforms

class CustomTensorDataset(Dataset):
    def __init__(self, data_X, data_y, transform_list=None):
        # X_tensor, y_tensor = torch.tensor(data_X), torch.LongTensor(data_y)
        tensors = (data_X, data_y)
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
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
        
        cifar10_stats = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        }
        transform = transforms.Compose([
            transforms.Normalize(cifar10_stats["mean"], cifar10_stats["std"]),
        ])
        dataset = CustomTensorDataset(tensor_x, tensor_y, transform_list=transform)  # create your dataset
        return dataset
    
    def get_train_data(self, u_id, num_batchs):
        data = [next(self.train_sets[u_id]) for _ in range(num_batchs)]
        return data
    
    def get_all_test_data(self, u_id):
        return self.test_sets[u_id]
