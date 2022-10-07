from typing import List

import torch

from blades.clients.client import BladesClient


class Fltrust(object):
    r"""``Fltrust`` it a trusted-based aggregator from paper `FLTrust:
    Byzantine-robust Federated Learning via Trust Bootstrapping.

    <https://arxiv.org/abs/2012.13995>`_.
    """

    def __call__(self, clients: List[BladesClient]):
        trusted_clients = [client for client in clients if client.is_trusted()]
        assert len(trusted_clients) == 1
        trusted_client = trusted_clients[0]

        untrusted_clients = [client for client in clients if not client.is_trusted()]
        trusted_update = trusted_client.get_update()
        trusted_norm = torch.norm(trusted_update).item()
        untrusted_updates = list(map(lambda w: w.get_update(), untrusted_clients))
        cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        ts = torch.Tensor(
            list(
                map(
                    lambda update: torch.nn.functional.relu(
                        cosine_similarity(trusted_update, update)
                    ),
                    untrusted_updates,
                )
            )
        )
        pseudo_gradients = torch.vstack(
            list(
                map(
                    lambda update: update * trusted_norm / torch.norm(update).item(),
                    untrusted_updates,
                )
            )
        )
        true_update = (pseudo_gradients.T @ ts) / ts.sum()
        return true_update
