from .autogm import Autogm
from .clippedclustering import Clippedclustering
from .clustering import Clustering
from .dnc import Dnc
from .fltrust import Fltrust
from .geomed import Geomed
from .mean import Mean
import importlib
from typing import Dict, Optional
from .median import Median
from .multikrum import Multikrum
from .signguard import Signguard
from .trimmedmean import Trimmedmean


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
