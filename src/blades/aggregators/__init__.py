import importlib
from typing import Dict, Optional
from typing import List, Union

import numpy as np
import torch

from blades.clients.client import BladesClient
from .autogm import Autogm
from .clippedclustering import Clippedclustering
from .clustering import Clustering
from .dnc import Dnc
from .fltrust import Fltrust
from .geomed import Geomed
from .mean import Mean
from .median import Median
from .multikrum import Multikrum
from .signguard import Signguard
from .trimmedmean import Trimmedmean


def _get_updates(inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]):
    if all(isinstance(element, BladesClient) for element in inputs):
        updates = torch.stack(list(map(lambda w: w.get_update(), inputs)))
    elif isinstance(inputs, List) and all(
        isinstance(element, torch.Tensor) for element in inputs
    ):
        updates = torch.stack(inputs, dim=0)
    else:
        updates = inputs
    return updates


def bucketing_wrapper(aggregator, bucketing=5):
    """Key functionality.

    Forked from link:
    https://github.com/epfml/byzantine-robust-noniid-optimizer/blob/main/utils.py
    """
    print("Using bucketing wrapper.")

    def aggr(inputs):
        inputs = _get_updates(inputs)
        indices = list(range(len(inputs)))
        np.random.shuffle(indices)
        n = len(inputs)
        T = int(np.ceil(n / bucketing))

        reshuffled_inputs = []
        for t in range(T):
            indices_slice = indices[t * bucketing: (t + 1) * bucketing]
            g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
            reshuffled_inputs.append(g_bar)
        return aggregator(reshuffled_inputs)

    return aggr


def get_aggregator(agg, agg_kws, *, bucketing=5):
    aggr = init_aggregator(agg, agg_kws)
    if bucketing == 0:
        return aggr

    return bucketing_wrapper(aggr, bucketing)


def init_aggregator(aggregator, aggregator_kws: Optional[Dict] = {}):
    if type(aggregator) == str:
        agg_path = importlib.import_module("blades.aggregators.%s" % aggregator)
        agg_scheme = getattr(agg_path, aggregator.capitalize())
        aggregator = agg_scheme(**aggregator_kws)
    else:
        aggregator = aggregator
    return aggregator


__all__ = [
    "Autogm",
    "Clippedclustering",
    "Clustering",
    "Dnc",
    "Geomed",
    "Fltrust",
    "Mean",
    "Median",
    "Multikrum",
    "Trimmedmean",
    "Signguard",
    "init_aggregator",
]
