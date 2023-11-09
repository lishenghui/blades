from typing import Dict

import torch

from fllib.algorithms import Algorithm
from fllib.constants import CLIENT_UPDATE
from .adversary import Adversary


def find_orthogonal_unit_vector(v):
    # 随机生成一个与v不平行的向量
    random_vector = torch.randn_like(v)

    # 使用Gram-Schmidt正交化方法
    orthogonal_vector = (
        random_vector - torch.dot(random_vector, v) / torch.dot(v, v) * v
    )

    # 标准化向量
    orthogonal_vector[-200] = 20000
    orthogonal_vector[-100] = 20000
    orthogonal_unit_vector = orthogonal_vector / torch.norm(orthogonal_vector)
    return orthogonal_unit_vector


class SignGuardAdversary(Adversary):
    def __init__(self, clients, global_config: Dict = None):
        super().__init__(clients, global_config)

    def on_local_round_end(self, algorithm: Algorithm):
        updates = self._attack_sign_guard(algorithm)
        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result["id"])
            if client.is_malicious:
                result[CLIENT_UPDATE] = updates

        return updates

    def _attack_sign_guard(self, algorithm: Algorithm):
        benign_updates = self.get_benign_updates(algorithm)
        device = benign_updates.device
        mean_grads = benign_updates.mean(dim=0)

        # l2norms = [torch.norm(update).item() for update in benign_updates]

        # base_vector = find_orthogonal_unit_vector(mean_grads)
        # return M * base_vector
        num_para = len(mean_grads)
        pos = (mean_grads > 0).sum().item()  # / num_para
        neg = (mean_grads < 0).sum().item()  # / num_para
        zeros = (mean_grads == 0).sum().item()  # / num_para
        noise = torch.hstack(
            [
                torch.rand(pos, device=device),
                -torch.rand(neg, device=device),
                # torch.ones(pos, device=device),
                # -torch.ones(neg, device=device),
                torch.zeros(zeros, device=device),
            ]
        )

        perm = torch.randperm(num_para)

        # Shuffle the vector based on the generated permutation
        update = noise[perm]
        return update
