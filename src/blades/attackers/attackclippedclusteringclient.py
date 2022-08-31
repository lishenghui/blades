import numpy as np
import torch
from numpy import inf
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering

from blades.aggregators.clippedclustering import Clippedclustering
from blades.client import ByzantineClient


class AttackclippedclusteringClient(ByzantineClient):
    r"""
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        
        

class AttackclippedclusteringAdversary():
    def __init__(self, linkage='single'):
        self.linkage = linkage
        self.agg = Clippedclustering()

    def chain_attack(self, simulator):
        benign_update = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                benign_update.append(w.get_update())
        benign_update = torch.stack(benign_update, 0)
        benign_mean = benign_update.mean(dim=0).cpu().detach().numpy()
    
        np_models = benign_update.cpu().detach().numpy()
    
        num = len(benign_update)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                if i == j:
                    dis_max[i, j] = 0
                else:
                    dis_max[i, j] = spatial.distance.cosine(np_models[i, :], np_models[j, :])
        dis_max[dis_max == -inf] = 0
        dis_max[dis_max == inf] = 2
        dis_max[np.isnan(dis_max)] = 2
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='single', n_clusters=2)
        clustering.fit(dis_max)
    
        dis_cross = inf
        for idx_i, label_i in enumerate(clustering.labels_):
            for idx_j, label_j in enumerate(clustering.labels_):
                if idx_j == idx_i:
                    continue
                dis = spatial.distance.cosine(np_models[idx_i, :], np_models[idx_j, :])
                dis_cross = min(dis_cross, dis)
    
        theta_cross = np.arccos(1 - dis_cross) - 0.1

        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        # values = torch.vstack(list(model for model, label in zip(updates, clustering.labels_) if label == flag)).mean( dim=0)
        larger_group = torch.vstack(list(model / torch.norm(model) for model, label in zip(benign_update, clustering.labels_) if label == flag)).cpu().detach().numpy()
        # theta_cross = 0
        dis2mean = [spatial.distance.cosine(benign_update, benign_mean) for benign_update in larger_group]
        idx_max_dis = np.argmax(dis2mean)
        theta = np.arccos(1 - dis2mean[idx_max_dis])
        mal_update = benign_update[idx_max_dis] / torch.norm(benign_update[idx_max_dis])
        print(dis_cross)
        for w in simulator.get_clients():
            if w.is_byzantine():
                if theta + theta_cross >= np.pi:
                    mal_update = -10 * benign_update.mean(dim=0)
                else:
                    a = (np.cos(theta + theta_cross - 1e-4) - np.sin(theta + theta_cross - 1e-4) / np.tan(theta))
                    b = (np.cos(theta_cross - 1e-4) + np.sin(theta_cross - 1e-4) / np.tan(theta))
                    mal_update = 10 * (a * benign_update.mean(dim=0) / torch.norm(benign_update.mean(dim=0)) + b * mal_update)
                    theta = theta + theta_cross - 1e-4
                w.save_update(mal_update)
                dis = spatial.distance.cosine(mal_update.cpu().detach().numpy(), benign_update.mean(dim=0).cpu().detach().numpy())
                print(dis)
        return
    
    def attack_average(self, simulator):
        benign_update = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                benign_update.append(w.get_update())
        benign_update = torch.stack(benign_update, 0)
        benign_mean = benign_update.mean(dim=0).cpu().detach().numpy()
    
        np_models = benign_update.cpu().detach().numpy()
    
        num = len(benign_update)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                if i == j:
                    dis_max[i, j] = 0
                else:
                    dis_max[i, j] = spatial.distance.cosine(np_models[i, :], np_models[j, :])
        dis_max[dis_max == -inf] = 0
        dis_max[dis_max == inf] = 2
        dis_max[np.isnan(dis_max)] = 2
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=2)
        clustering.fit(dis_max)
    
        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        # values = torch.vstack(list(model for model, label in zip(updates, clustering.labels_) if label == flag)).mean( dim=0)
        larger_group = torch.vstack(
            list(model / torch.norm(model) for model, label in zip(benign_update, clustering.labels_) if label == flag))
        center_point = torch.mean(larger_group, dim=0)
    
        dis_cross = 0
        dis_size = 0
        for i in range(len(clustering.labels_)):
            for j in range(i + 1, len(clustering.labels_)):
                if clustering.labels_[i] != clustering.labels_[j]:
                    dis = spatial.distance.cosine(np_models[i, :], np_models[j, :])
                    dis_cross += dis
                    dis_size += 1
    
        dis_avg = dis_cross / dis_size
        print('avg dis', dis_avg)
        theta_cross = np.arccos(1 - dis_avg) - 0.1
        theta = np.arccos(1 - spatial.distance.cosine(center_point, benign_mean))
        print(dis_cross)
        for w in simulator.get_clients():
            if w.is_byzantine():
                if theta + theta_cross >= np.pi:
                    mal_update = -1000 * benign_update.mean(dim=0)
                else:
                    a = (np.cos(theta + theta_cross) - np.sin(theta + theta_cross) / np.tan(theta))
                    b = (np.cos(theta_cross) + np.sin(theta_cross) / np.tan(theta))
                    mal_update = 1000 * (a * benign_update.mean(dim=0) / torch.norm(benign_update.mean(dim=0)) + b * center_point / torch.norm(center_point))
                w.save_update(mal_update)
                dis = spatial.distance.cosine(mal_update.cpu().detach().numpy(),
                                              center_point.cpu().detach().numpy())
                print(np.arccos(1 - dis))
        return
    
    def omniscient_callback(self, simulator):
        if self.linkage == "single":
            return self.chain_attack(simulator)
        elif self.linkage == "average":
            return self.attack_average(simulator)
        else:
            raise NotImplementedError(f"linkage {self.linkage} is not implemented.")
