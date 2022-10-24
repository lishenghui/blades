import importlib
from .labelflippingclient import LabelflippingClient
from .signflippingclient import SignflippingClient
from .alieclient import AlieClient
from .noiseclient import NoiseClient
from .ipmclient import IpmClient
from typing import Dict, Optional


def init_attacker(attack, attack_kws: Optional[Dict] = {}):
    if type(attack) == str:
        # atk_path = importlib.import_module(".")
        atk_path = importlib.import_module("blades.attackers.%s" % attack + "client")
        atk_scheme = getattr(atk_path, attack.capitalize() + "Client")
        client = atk_scheme(**attack_kws)
    else:
        client = client
    return client


__all__ = [
    "NoiseClient",
    "AlieClient",
    "IpmClient",
    "SignflippingClient",
    "LabelflippingClient",
]
