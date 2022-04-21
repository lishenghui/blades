import numpy as np
import ray
from math import ceil

from .fedmodel import FedModel


class ActorManager():
    def __init__(self, params, model):
        for key, val in vars(params).items(): setattr(self, key, val)
        if model is not None:
            self.RemoteModel = ray.remote(num_gpus=self.gpu_per_actor)(model)
        else:
            self.RemoteModel = ray.remote(num_gpus=self.gpu_per_actor)(FedModel)
    
    def init_actors(self, net):
        self.actors = [self.RemoteModel.remote(net, self.seed, *self.model_params) for _ in range(self.num_actors)]
        print("----Initialized %d actors------" % self.num_actors)
        global_model = ray.get(self.actors[0].get_params.remote())
        self.set_parameters_all(global_model)
        return global_model
    
    def set_parameters_all(self, params):
        ray.get([actor.set_params.remote(params) for actor in self.actors])
    
    def set_parameters_multi_actors(self, id_list, client_list):
        ray.get([self.actors[id].set_params.remote(client.local_model) for id, client in zip(id_list, client_list)])
    
    def vote(self, clients, cur_net, pre_net, **kwargs):
        result = []
        actor_size = len(self.actors)
        for client_batch in np.array_split(clients, ceil(len(clients) / actor_size)):
            params = ray.get(
                [client.vote(self.actors[i], cur_net, pre_net, **kwargs) for i, client in
                 enumerate(client_batch)])
            result.extend(params)
        return result
    
    def train(self, clients, model_params, scale=False, **kwargs):
        updates = []
        actor_size = len(self.actors)
        for client_batch in np.array_split(clients, ceil(len(clients) / actor_size)):
            self.set_parameters_all(model_params)
            params = ray.get(
                [client.train(self.actors[i], self.local_rounds, model_params, scale=scale, **kwargs) for i, client in
                 enumerate(client_batch)])
            updates.extend(params)
        # for idx, client in enumerate(clients):
        #     if client.colluding:
        #         updates[idx][3] = copy.deepcopy(kwargs['init_model'])
        return updates
    
    def test(self, clients, model_params=None):
        actor_size = len(self.actors)
        metrics = []
        for client_batch in np.array_split(clients, ceil(len(clients) / actor_size)):
            if model_params:
                self.set_parameters_all(model_params)
            else:
                self.set_parameters_multi_actors(list(range(len(client_batch))), client_batch)
            params = ray.get([client.test(self.actors[i]) for i, client in enumerate(client_batch)])
            metrics.extend(params)
        
        return {'loss': np.around(np.average([d['loss'] for d in metrics], weights=[d['size'] for d in metrics]), 3),
                'accuracy': np.around(
                    np.average([d['accuracy'] for d in metrics], weights=[d['size'] for d in metrics]), 3),
                'individual_accs': [(d['loss'], d['entropy'], d['accuracy'], d['size']) for d in metrics]
                }
