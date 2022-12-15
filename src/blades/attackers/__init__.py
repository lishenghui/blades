import importlib
from typing import Dict, Optional

from .alieclient import AlieClient
from .ipmclient import IpmClient
from .labelflippingclient import LabelflippingClient
from .noiseclient import NoiseClient
from .signflippingclient import SignflippingClient


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
