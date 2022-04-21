import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import linalg as LA


class FedModel():
    def __init__(self, net, seed, num_classes, lr, test_bs=128):
        self.lr = lr
        self.test_bs = test_bs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_random_seed(seed)
        self.net = net(num_classes).to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.net = nn.DataParallel(self.net)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=0.01)
    
    def set_params(self, model_params):
        self.net.load_state_dict(copy.deepcopy(model_params))
    
    def save_model(self, path='./'):
        torch.save(self.net, path)
    
    def get_params(self):
        return self.net.state_dict()
    
    def set_random_seed(self, seed_value):
        torch.manual_seed(seed_value)  # cpu  vars
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # gpu vars
            torch.backends.cudnn.deterministic = True  # needed
            torch.backends.cudnn.benchmark = False
    
    def update(self, dataset, epochs, g_net, l_net, noise, scale, **kwargs):
        if kwargs['attack_type'] == 'colluding':
            return copy.deepcopy(kwargs['init_model']), 0.0
        elif kwargs['attack_type'] == 'free_rider':
            return copy.deepcopy(g_net), 0.0
        
        # if kwargs['val'] == 'val':
        #     return self.update_valid(dataset, epochs, g_net, l_net, noise, scale, **kwargs)
        curr_global_model = copy.deepcopy(g_net)
        
        # valid_set = kwargs['valid_set']
        # loss_hist = self.test_model(valid_set)["loss"]
        # curr_local_model = copy.deepcopy(l_net)
        self.net.train()
        
        self.set_params(curr_global_model)
        for i, (data, target) in enumerate(dataset):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target) * scale
            loss.backward()
            # for name, param in self.net.named_parameters():
            #     if param.grad is not None:
            #         param.grad += kwargs['lamb'] * (self.get_params()[name] - curr_global_model[name])
            
            if i <= 5:
                pass
                # gradient_norm = LA.norm(torch.tensor([LA.norm(p.grad.data) for name, p in self.net.named_parameters()]))
                # print(i, kwargs['attack_type'], loss.item(), gradient_norm)
                # print(i, kwargs['attack_type'], {name : '%.3f' % LA.norm(p.grad.data).item() for name, p in self.net.named_parameters()})
            if kwargs['attack_type'] == 'sign_flipping':
                for name, p in self.net.named_parameters():
                    p.grad.data = -p.grad.data
            
            if kwargs['attack_type'] == 'scaling':
                for name, p in self.net.named_parameters():
                    p.grad.data *= 100
            
            if kwargs['clip']:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), kwargs['clip'])
            self.optimizer.step()
        curr_global_model = copy.deepcopy(self.get_params())
        
        for k in curr_global_model.keys():
            curr_global_model[k] -= g_net[k]
        
        # if kwargs['attack_type'] == 'scaling':
        #         for k in curr_global_model.keys():
        #             curr_global_model[k] *= 50
        
        if noise > 0:
            for k in curr_global_model.keys():
                curr_global_model[k] = torch.ones_like(curr_global_model[k]).to(self.device) / 3
                # curr_global_model[k] += torch.normal(noise, noise / 100, size=curr_global_model[k].shape).to(self.device)
        return curr_global_model, 0
    
    def update_valid(self, dataset, epochs, g_net, l_net, noise, scale, **kwargs):
        if kwargs['attack_type'] == 'colluding':
            return None, None, copy.deepcopy(kwargs['init_model']), copy.deepcopy(kwargs['init_model'])
        elif kwargs['attack_type'] == 'free_rider':
            return None, None, copy.deepcopy(l_net), copy.deepcopy(g_net)
        
        g_net_pre = copy.deepcopy(kwargs['pre_global_model'])
        g_net_cur = copy.deepcopy(g_net)
        valid_set = kwargs['valid_set']
        self.set_params(g_net_cur)
        valid_loss_cur = self.test_model(valid_set)["loss"]
        self.set_params(g_net_pre)
        valid_loss_last = self.test_model(valid_set)["loss"]
        print(valid_loss_cur, valid_loss_last)
        if valid_loss_last > valid_loss_cur:
            self.set_params(g_net_cur)
            num_rounds = len(dataset)
        else:
            print("Round: %d, pred_loss: %.3f, curr_loss: %.3f, global model rejected." % (
                kwargs['round'], valid_loss_last, valid_loss_cur))
            self.set_params(g_net_pre)
            num_rounds = len(dataset)
        
        self.net.train()
        for data, target in dataset[:num_rounds]:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target) * scale
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), kwargs['clip'])
            self.optimizer.step()
        
        curr_global_model = copy.deepcopy(self.get_params())
        
        if noise > 0:
            for k in curr_global_model.keys():
                curr_global_model[k] += torch.normal(noise, noise / 100,
                                                     size=curr_global_model[k].shape).to(self.device)
        
        return curr_global_model
    
    def update_valid_back(self, dataset, epochs, g_net, l_net, noise, scale, **kwargs):
        if kwargs['attack_type'] == 'colluding':
            return None, None, copy.deepcopy(kwargs['init_model']), copy.deepcopy(kwargs['init_model'])
        elif kwargs['attack_type'] == 'free_rider':
            return None, None, copy.deepcopy(l_net), copy.deepcopy(g_net)
        
        curr_local_model = copy.deepcopy(l_net)
        g_net_pre = copy.deepcopy(kwargs['pre_global_model'])
        g_net_cur = copy.deepcopy(g_net)
        valid_set = kwargs['valid_set']
        self.set_params(g_net_cur)
        valid_loss_cur = self.test_model(valid_set)["loss"]
        self.set_params(g_net_pre)
        valid_loss_last = self.test_model(valid_set)["loss"]
        
        if valid_loss_last > valid_loss_cur - 0.1:
            g_net = copy.deepcopy(g_net_cur)
            self.set_params(g_net_cur)
            num_rounds = len(dataset)
        else:
            print("Round: %d, pred_loss: %.3f, curr_loss: %.3f, global model rejected." % (
                kwargs['round'], valid_loss_last, valid_loss_cur))
            g_net = copy.deepcopy(g_net_pre)
            num_rounds = len(dataset)
        
        self.net.train()
        for data, target in dataset[:num_rounds]:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target) * scale
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), kwargs['clip'])
            self.optimizer.step()
        
        curr_global_model = copy.deepcopy(self.get_params())
        
        self.set_params(curr_local_model)
        self.net.train()
        for data, target in dataset[:num_rounds]:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target) * scale
            loss.backward()
            if kwargs['lamb'] > 0:
                for name, param in self.net.named_parameters():
                    if param.grad is not None:
                        param.grad += kwargs['lamb'] * (self.get_params()[name] - g_net[name])
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), kwargs['clip'])
            self.optimizer.step()
        curr_local_model = copy.deepcopy(self.get_params())
        
        if noise > 0:
            for k in curr_global_model.keys():
                curr_global_model[k] += torch.normal(noise, noise / 100,
                                                     size=curr_global_model[k].shape).to(self.device)
        
        return [None, None, curr_local_model, curr_global_model]
    
    def test_model(self, test_loader):
        return self.net.test_model(test_loader, self.device)
