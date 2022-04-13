import json
import os
import pickle
import sys

utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)


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
    
    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)
    
    with open(os.path.join(os.path.split(train_data_dir)[0], 'data_cache.obj'), 'wb') as f:
        pickle.dump(train_clients, f)
        pickle.dump(train_data, f)
        pickle.dump(test_clients, f)
        pickle.dump(test_data, f)
    
    assert sorted(train_clients) == sorted(test_clients)
    assert train_groups == test_groups
    
    return train_clients, train_groups, train_data, test_data


read_data('../train', '../test')
