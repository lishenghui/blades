import json
import os
import pickle

import numpy as np
import torch
from scipy.sparse import csr_matrix
from time import time
import math

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    }
}
img_mean = DATASETS['cifar10']['mean']
img_std = DATASETS['cifar10']['std']
img_size = DATASETS['cifar10']['img_size']

augmentations = []
# from utils.autoaug import CIFAR10Policy
# augmentations += [CIFAR10Policy()]
augmentations += [
    transforms.RandomCrop(img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=img_mean, std=img_std)
]

test_augmentations = transforms.Compose(augmentations)


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)

    if not isinstance(data_x, csr_matrix):
        np.random.shuffle(data_x)
        np.random.shuffle(data_y)
        sz = len(data_x)
    else:
        sz = data_x.shape[0]
    # loop through mini-batches
    for i in range(0, sz, batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def preprocess_data(data, labels, batch_size, seed=0):
    i = 0
    np.random.seed(seed=seed)
    idx = np.random.permutation(len(labels))
    data, labels = data[idx], labels[idx]

    DATASETS = {
        'cifar10': {
            'num_classes': 10,
            'img_size': 32,
            'mean': [0.4914, 0.4822, 0.4465],
            "std": (0.2023, 0.1994, 0.2010)
        }
    }
    img_mean = DATASETS['cifar10']['mean']
    img_std = DATASETS['cifar10']['std']
    img_size = DATASETS['cifar10']['img_size']

    augmentations = []
    # from utils.autoaug import CIFAR10Policy
    # augmentations += [CIFAR10Policy()]
    augmentations += [
        # transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ]

    augmentations = transforms.Compose(augmentations)
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


def read_dir(data_dir):
    clients = []
    groups = []
    data = {}

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    return clients, groups, data


def read_data(data_path):
    cache_path = data_path
    assert os.path.isfile(cache_path)
    with open(cache_path, 'rb') as f:
        (train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
    train_groups = []
    
    assert sorted(train_clients) == sorted(test_clients)

    return train_clients, train_groups, train_data, test_data
