import numpy as np
from numpy import linalg as LA

from scipy.spatial.distance import cdist, euclidean


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)
    
    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]
        
        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)
        
        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y
        
        if euclidean(y, y1) < eps:
            return y1
        
        y = y1


class Client:
    def __init__(self, parameter):
        self.num_train_samples = 1
        self.parameter = np.array(parameter)


def aggregate(update_list, alphas=None):
    weights = alphas
    nor_weights = np.array(weights) / np.sum(weights)
    avg_updates = np.sum(np.stack([param * weight for param, weight in zip(update_list, nor_weights)]), axis=0)
    
    return avg_updates


def l2dist(model1, model2):
    return LA.norm(model1 - model2)

def l2norm(model):
    return LA.norm(model)

def geometric_median_objective(median, points, alphas):
    return sum([alpha * l2dist(median, p) for alpha, p in zip(alphas, points)])


def weighted_average_oracle(points, weights):
    tot_weights = np.sum(weights)
    weighted_updates = [np.zeros_like(v) for v in points[0]]
    
    for w, p in zip(weights, points):
        for j, weighted_val in enumerate(weighted_updates):
            weighted_val += (w / tot_weights) * p[j]
    
    return weighted_updates


def auto_gm(clients, lamb=1.0, maxiter=100, eps=1e-5, ftol=1e-6):
    param_list = [client.parameter for client in clients]
    lamb = lamb * (len(clients))
    alpha = np.ones(shape=len(clients)) / len(clients)
    global_model = None
    for i in range(maxiter):
        median = aggregate(param_list, alpha)
        obj_val = geometric_median_objective(median, param_list, alpha)
        global_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray(
                [alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alpha, param_list)],
                dtype=alpha.dtype)
            weights = weights / weights.sum()
            median = aggregate(param_list, weights)
            obj_val = geometric_median_objective(median, param_list, alpha)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        
        global_model = median
        for client in clients:
            client.distance = l2dist(median, client.parameter)
        
        # Update weights
        idxs = [x for x, _ in sorted(enumerate(clients), key=lambda x: x[1].distance)]
        eta_optimal = clients[idxs[0]].distance + lamb / clients[idxs[0]].num_train_samples
        for p in range(0, len(idxs)):
            eta = (sum([clients[i].distance for i in idxs[:p + 1]]) + lamb) / (p + 1)
            if p < len(idxs) and eta - clients[idxs[p]].distance < 0:
                break
            else:
                eta_optimal = eta
        alpha = np.array([max(eta_optimal - c.distance, 0) / lamb for c in
                          clients])
        
        new_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
        if abs(new_obj - global_obj) < ftol * new_obj:
            break
    print(global_obj)
    return alpha, global_model


if __name__ == "__main__":
    clients = [Client([1, 2]) for _ in range(2)]
    _, gm = auto_gm(clients)
    print(gm)
