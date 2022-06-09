import torch
from blades.client import BladesClient
from blades.server import BladesServer
from typing import Any, Callable, Optional, Union, List

class _FltrustedAGG(object):
    
    def __call__(self, clients: List[BladesClient]):
        trusted_clients = [client for client in clients if client.get_is_trusted()]
        assert len(trusted_clients) == 1
        trusted_client = trusted_clients[0]
        
        untrusted_clients =[client for client in clients if not client.get_is_trusted()]
        trusted_update = trusted_client.get_update()
        untrusted_updates = list(map(lambda w: w.get_update(), untrusted_clients))
        cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        ts = list(map(lambda update:
                     torch.nn.functional.relu(
                         cosine_similarity(trusted_update, update)
                     ),
                    untrusted_updates,
                    )
             )
        return ts


class FltrustServer(BladesServer):
    def __int__(self, *args, **kwargs):
        super(FltrustServer, self).__int__(*args, **kwargs)
    
    
