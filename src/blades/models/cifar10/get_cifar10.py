import argparse
import os
import pickle

import numpy as np
import torchvision
from sklearn.utils import shuffle


def generate_datasets(iid=False, alpha=1.0, num_clients=100):
    train_set = torchvision.datasets.CIFAR10(train=True, download=True, root="./data")
    test_set = torchvision.datasets.CIFAR10(train=False, download=True, root="./data")
    x_test, y_test = test_set.data, np.array(test_set.targets)
    x_train, y_train = train_set.data, np.array(train_set.targets)
    
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
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
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
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
    
    #     os.system('rm -rf ..')
    #     os.system('mkdir -p ../data')
    with open(os.path.join('data', 'data_cache' + ("_alpha" + str(args.alpha) if not args.iid else "") + '.obj'),
              'wb') as f:
        pickle.dump(train_user_ids, f)
        pickle.dump(train_dataset, f)
        pickle.dump(train_user_ids, f)
        pickle.dump(test_dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--iid', dest='iid', action='store_true')
    flag_parser.add_argument('--noniid', dest='iid', action='store_false')
    parser.add_argument('--alpha', type=float, default=0.1, required=False)
    parser.add_argument('--num_clients', type=int, default=20, required=False)
    parser.set_defaults(iid=True)
    args = parser.parse_args()
    generate_datasets(args.iid, alpha=args.alpha, num_clients=args.num_clients)
