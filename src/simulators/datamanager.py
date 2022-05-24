import os
import pickle

import numpy as np

from settings.data_utils import load_data
from settings.data_utils import preprocess_data


class DataManager(object):
    def __init__(self, data_path, batch_size, *args, **kwargs):
        self.train_sets = {}
        self.test_sets = {}
        assert os.path.isfile(data_path)
        with open(data_path, 'rb') as f:
            (train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
        
        assert sorted(train_clients) == sorted(test_clients)
        self.clients = train_clients
        for idx, u_id in enumerate(train_clients):
            self.train_sets[u_id] = preprocess_data(np.array(train_data[u_id]['x']), np.array(train_data[u_id]['y']),
                                                    batch_size=batch_size)
            self.test_sets[u_id] = load_data(train_data[u_id])
    
    def get_data(self, u_id, num_batchs):
        data = [next(self.train_sets[u_id]) for _ in range(num_batchs)]
        return data
