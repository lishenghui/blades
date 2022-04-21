import copy
import logging
import numpy as np
import os
import random
import time
import torch
from abc import ABCMeta, abstractmethod
from torch import linalg as LA
from torch.utils.data import TensorDataset, DataLoader
from utils.data_utils import read_data, preprocess_data

from models.actormanager import ActorManager


class BaseClient:
    
    def __init__(self, client_id, train_data, test_data, seed=0, attack=False, attack_type=0):
        self.id = client_id
        self.is_corrupted = False
        self.attack_type = -1
        self.loss = {}
        self.raw_train_data = train_data
        self.raw_test_data = test_data
        self.seed = seed
        self.noise = 0
        self.scale = 1.0
        self.free_rider = False
        self.colluding = False
        self.valid_data = None
        self.best_global = None
        self.his_loss = []
    
    def train(self, model, epochs, g_net, scale=False, **kwargs):
        data = [next(self.train_data) for _ in range(epochs)]
        scale_val = self.scale if scale else 1.0
        
        return model.update.remote(data, epochs, g_net, self.local_model, attack_type=self.attack_type,
                                   noise=self.noise, scale=scale_val, **dict(kwargs, valid_set=self.valid_data,
                                                                             g_net_pre=self.best_global))
    
    def vote(self, model, cur_net, pre_net, **kwargs):
        return model.vote.remote(cur_net, pre_net, self.valid_data,
                                 **dict(kwargs, his_loss=self.his_loss, is_corrupted=self.is_corrupted))
    
    def set_datasets(self, batch_size=32, valid=False):
        valid_rate = 0.0 if valid else 0.0
        train_len = len(self.raw_train_data['y'])
        indices = list(range(train_len))
        random.shuffle(indices)
        
        train_indices = indices[int(train_len * valid_rate):]
        train_x = np.array(self.raw_train_data['x'])[train_indices, :]
        train_y = np.array(self.raw_train_data['y'])[train_indices]
        self.train_data = preprocess_data(train_x, train_y, batch_size=batch_size)
        
        # if valid:
        #     valid_indices = indices[: int(train_len * valid_rate)]
        #     valid_x = np.array(self.raw_train_data['x'])[valid_indices, :]
        #     valid_y = np.array(self.raw_train_data['y'])[valid_indices]
        #     self.valid_data = self._load_data(valid_x, valid_y, batch_size=batch_size)
        if valid:
            valid_indices = indices[:]
            valid_x = np.array(self.raw_train_data['x'])[valid_indices, :]
            valid_y = np.array(self.raw_train_data['y'])[valid_indices]
            self.valid_data = self._load_data(valid_x, valid_y, batch_size=batch_size)
        
        self.test_data = self._load_data(self.raw_test_data['x'], self.raw_test_data['y'])
    
    def _load_data(self, data_x, data_y, batch_size=32):
        tensor_x = torch.Tensor(data_x)  # transform to torch tensor
        tensor_y = torch.LongTensor(data_y)
        
        dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        dataloader = DataLoader(dataset, batch_size=batch_size)  # create your dataloader
        return dataloader
    
    def poison_data(self, p, L):
        
        """
            poison_by_shuffling_labels selects a fraction p of the samples and
            shuffles their labels randomly
        """
        self.attack_type = 'data'
        sz = len(self.raw_train_data['y'])
        self.raw_train_data['y'] = np.array(self.raw_train_data['y'])
        n_poisoned = int(sz * p)
        poisoned_points = np.random.choice(sz, n_poisoned, replace=False)
        reordered = np.random.permutation(poisoned_points)
        # self.raw_train_data['y'][poisoned_points] = self.raw_train_data['y'][reordered]
        self.raw_train_data['y'][poisoned_points] = L - self.raw_train_data['y'][poisoned_points] - 1
        
        return self.raw_train_data
    
    def poison_model(self, max_noise=0.1, colluding=False):
        self.attack_type = 'model'
        if colluding:
            self.noise = 0.05
        else:
            self.noise = np.random.uniform(0.05, max_noise)
    
    def poison_by_noise(self, args):
        
        x = np.array(self.train_data['x'])
        x_new = x
        if args.dataset == 'femnist':
            scale = 0.7
            noise = np.random.normal(0, scale, x.shape)
            x_noisy = x + noise
            x_new = (x_noisy - np.min(x_noisy)) / (np.max(x_noisy) - np.min(x_noisy))
            
            # img = np.array(x_new[0]).reshape((28, 28))
            # plt.imshow(img, cmap='gray', aspect='equal')
            # plt.grid(False)
            # _ = plt.show()
        # modify client in-place
        self.train_data['x'] = x_new
    
    def test(self, model):
        return model.test_model.remote(self.test_data)
    
    @property
    def num_test_samples(self):
        if self.test_data is None:
            return 0
        return len(self.test_data.dataset)
    
    @property
    def num_train_samples(self):
        if self.train_data is None:
            return 0
        return len(self.raw_train_data['y'])
    
    @property
    def num_samples(self):
        return self.num_train_samples + self.num_test_samples


class BaseServer(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, params, net, dataset, client, model=None, num_class=10):
        self.dataset = dataset
        for key, val in vars(params).items(): setattr(self, key, val)
        self.clients = self.setup_clients(client)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('{} Clients in Total'.format(len(self.clients)))
        self.model = model(net, 0, num_class, 0.01, 64)
        self.train_proxy = ActorManager(params, model)
        self.global_model = self.train_proxy.init_actors(net)
        self.init_global_model = copy.deepcopy(self.global_model)
        self.corrupted_counter = 0
        
        self.dis_history = []
        self.time_begin = time.time()
        
        # self.norm_clipping = None
        self.corrupt_clients(trusted_idx=self.trusted)
        logging.basicConfig(filename=self.log_filename, level=logging.INFO, format='%(message)s')
    
    def l2dist(self, model1, model2, ratio=1.0):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        return LA.norm(torch.tensor([LA.norm(model1[k] - model2[k]) for k in list(model1.keys())]))
    
    def cos_sim(self, model1, model2):
        return torch.sum(torch.tensor([torch.sum(model1[k] * model2[k]) for k in model1])) / torch.max(
            self.l2norm(model1) * self.l2norm(model2), torch.tensor(0.00001))
        # return torch.sum([torch.dot(model1[k], model2[k]) for k in model1])# / torch.max(self.l2norm(model1) * self.l2norm(model2), 0.00001)
    
    def l2norm(self, model):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        return LA.norm(torch.tensor([LA.norm(model[k]) for k in model]))
    
    def clipping_filter(self, parameter_list, clients, max_norm):
        for para, client in zip(parameter_list, clients):
            pass
            # print(self.l2dist(para, self.global_model) ** 2, client.is_corrupted, max_norm)
        # self.dis_history.extend([self.(update).item()  for update in parameter_list])
        self.dis_history.extend([self.l2dist(para, self.last_update).item() for para in parameter_list])
        # threshold = np.percentile(self.dis_history, 50)
        threshold = np.median([self.l2norm(para).item() for para in parameter_list])
        # threshold = np.median(self.dis_history)
        print(threshold)
        # threshold = np.median([self.l2dist(para, self.global_model).item()  for para in parameter_list])
        
        model_list = []
        client_list = []
        for update, client in zip(parameter_list, clients):
            # self.para_queue.put(update)
            # if client.is_corrupted:
            # clip_para_norm_(update, threshold)
            # if self.l2dist(update, self.last_update)  <= threshold:
            model_list.append(update)
            client_list.append(client)
            # else:
            # if self.last_update:
            # print('l2 norm with last update', self.l2dist(update, self.last_update).item(), client.is_corrupted)
            # clip_para_norm_(update, threshold)
            # print('l2 norm after', self.l2norm(update).item())
            # model_list.append(update)
            # client_list.append(client)
        # after_filter = zip(*((model, client) for model, client in zip(parameter_list, clients) if self.l2dist(model, self.global_model)  <= threshold))
        # print(len(after_filter))
        return model_list, client_list
    
    def select_clients(self, num_clients=20, weighted=False, trusted_idx=None):
        if trusted_idx:
            num_clients -= 1
        possible_clients = np.array([client for i, client in enumerate(self.clients) if i != trusted_idx])
        num_clients = min(num_clients, len(possible_clients))
        candidates = [i if i < trusted_idx else i + 1 for i in range(len(possible_clients))]
        print('candidates:', candidates)
        if weighted:
            p = np.array([client.num_train_samples for client in self.clients]) / \
                np.sum([client.num_train_samples for client in self.clients])
            selected_index = np.random.choice(candidates, num_clients, p=p, replace=False)
        else:
            selected_index = np.random.choice(candidates, num_clients, replace=False)
        
        if trusted_idx:
            self.selected_clients = np.insert(np.array(possible_clients)[selected_index], 0, self.clients[trusted_idx])
        else:
            self.selected_clients = np.array(possible_clients)[selected_index]
        
        print([client.id for client in self.selected_clients])
        corrupted_clients = [client for client in self.selected_clients if client.is_corrupted]
        print("%d corrupted clients are selected" % len(corrupted_clients))
        return self.selected_clients
    
    def corrupt_clients(self, trusted_idx):
        # Randomly attack clients
        pc = self.pc
        ps = self.ps
        att_type = self.attack
        n = int(len(self.clients) * pc)
        np.random.seed(self.seed)
        
        possible_clients = np.array([i for i, client in enumerate(self.clients) if i != trusted_idx])
        selected_indexes = np.random.choice(possible_clients, n, replace=False)
        selected_indexes = np.sort(selected_indexes)
        
        self.clients[trusted_idx].is_trusted = True
        for i in selected_indexes:
            self.clients[i].is_corrupted = True
            self.clients[i].attack_type = att_type
            if att_type == 'data':
                self.clients[i].poison_data(ps, 62)
            elif att_type == 'model':
                self.clients[i].poison_model(max_noise=0.5, colluding=True)
            elif att_type == 'free_rider':
                self.clients[i].free_rider = True
            elif att_type == 'colluding':
                self.clients[i].colluding = True
        print("attacked clients: " + ','.join([str(i) for i in selected_indexes]))
    
    def setup_clients(self, Client):
        train_data_dir = os.path.abspath(os.path.join('data', self.dataset, 'train'))
        test_data_dir = os.path.abspath(os.path.join('data', self.dataset, 'test'))
        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
        print(users)
        self.clients = []
        for i, u in enumerate(users):
            client = Client(u, train_data[u], test_data[u])
            self.clients.append(client)
        
        return self.clients
    
    @abstractmethod
    def run(self):
        pass
    
    def save(self, path='./model.pt'):
        # for key in list(self.global_model.keys()):
        # self.global_model['module.' + key] = self.global_model.pop(key)
        self.model.set_params(self.global_model)
        self.model.save_model(path)
    
    def test(self, round, model=None, loss_list=None):
        # Test model
        if round == 1 or round % self.eval_every == 0 or round == self.global_rounds:
            
            # metrics = self.train_proxy.test([client for client in self.clients], model)
            benign_clients = [client for client in self.clients if not client.is_corrupted]
            metrics = self.train_proxy.test(benign_clients, model)
            metrics['round'] = round
            metrics['train_liss'] = np.mean(loss_list)
            
            logging.info(metrics)
            if round >= 100 and metrics['accuracy'] < 30.0 and self.dataset == 'femnist':
                self.corrupted_counter += 1
                if self.corrupted_counter >= 30000:
                    print("Broken down!")
                    exit()
            else:
                self.corrupted_counter = 0
            print('------Current round %d, time cost %.2f------' % (round, time.time() - self.time_begin))
            print(metrics)
