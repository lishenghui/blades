import torch
import numpy as np
from blades.client import ByzantineClient
from blades.aggregators.clippedclustering import Clippedclustering
from sklearn.cluster import AgglomerativeClustering
from numpy import inf
from scipy import spatial
from blades.utils import torch_utils
class AttackclippedclusteringClient(ByzantineClient):
    r"""
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        
        

class AttackclippedclusteringAdversary():
    def __int__(self):
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
        clustering = AgglomerativeClustering(affinity='precomputed', linkage='complete', n_clusters=2)
        clustering.fit(dis_max)
    
        dis_cross = inf
        for idx_i, label_i in enumerate(clustering.labels_):
            for idx_j, label_j in enumerate(clustering.labels_):
                if idx_j == idx_i:
                    continue
                dis = spatial.distance.cosine(np_models[idx_i, :], np_models[idx_j, :])
                dis_cross = min(dis_cross, dis)
    
        theta_cross = np.arccos(1 - dis_cross) - 0.1
        # theta_cross = 0
        dis2mean = [spatial.distance.cosine(benign_update, benign_mean) for benign_update in np_models]
        idx_max_dis = np.argmax(dis2mean)
        theta = np.arccos(1 - dis2mean[idx_max_dis])
        mal_update = benign_update[idx_max_dis] / torch.norm(benign_update[idx_max_dis])
        print(dis_cross)
        for w in simulator.get_clients():
            if w.is_byzantine():
                if theta + theta_cross >= np.pi:
                    mal_update = -benign_update.mean(dim=0)
                    w.save_update(mal_update)
                else:
                    a = (np.cos(theta + theta_cross - 1e-4) - np.sin(theta + theta_cross - 1e-4) / np.tan(theta))
                    b = (np.cos(theta_cross - 1e-4) + np.sin(theta_cross - 1e-4) / np.tan(theta))
                    mal_update0 = a * benign_update.mean(dim=0) / torch.norm(benign_update.mean(dim=0)) + b * mal_update
                    dis = spatial.distance.cosine(mal_update0.cpu().detach().numpy(), mal_update.cpu().detach().numpy())
                    mal_update = mal_update0
                    theta = theta + theta_cross - 1e-4
                    w.save_update(mal_update)
        return

    def omniscient_callback(self, simulator):
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
        larger_group = torch.vstack(list(model / torch.norm(model) for model, label in zip(benign_update, clustering.labels_) if label == flag))
        center_point = torch.sum(larger_group, dim=0)
        
        
        dis_cross = 0
        dis_size = 0
        for i in range(len(clustering.labels_)):
            for j in range(i+1, len(clustering.labels_)):
                if clustering.labels_[i] != clustering.labels_[j]:
                    dis = spatial.distance.cosine(np_models[i, :], np_models[j, :])
                    dis_cross += dis
                    dis_size += 1
        
        dis_avg =  dis_cross / dis_size
        threshold = min(dis_size * (1 - dis_avg) / torch.norm(center_point), 1)
        # max_dis = np.arccos(1 - dis_cross / dis_size) * 15 / torch.norm(center_point)
        theta_cross = np.arccos(threshold)
        # theta_cross = 0
        dis2mean = [spatial.distance.cosine(benign_update, benign_mean) for benign_update in np_models]
        idx_max_dis = np.argmin(dis2mean)
        # theta = np.arccos(1 - dis2mean[idx_max_dis])
        theta = np.arccos(1 - spatial.distance.cosine(center_point, benign_mean))
        mal_update = center_point / torch.norm(center_point)
        print(dis_cross)
        for w in simulator.get_clients():
            if w.is_byzantine():
                if theta + theta_cross >= np.pi:
                    mal_update = -benign_update.mean(dim=0)
                    w.save_update(mal_update)
                else:
                    a = (np.cos(theta + theta_cross - 1e-4) - np.sin(theta + theta_cross - 1e-4) / np.tan(theta))
                    b = (np.cos(theta_cross - 1e-4) + np.sin(theta_cross - 1e-4) / np.tan(theta))
                    mal_update0 = a * benign_update.mean(dim=0) / torch.norm(benign_update.mean(dim=0)) + b * mal_update
                    w.save_update(mal_update0 * 5.0)
        return