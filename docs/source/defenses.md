
# Defenses

The following defense strategies are currently implemented in ``Blades``:
## Build-in Attacks

### Robust Aggregation

| Methods   | Descriptions                                                                                                                               | Examples                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **Krum**   | *Blanchard et al.* [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html), *NIPS'17*              | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/krum.py)   |
| **GeoMed**   | *Chen et al.* [Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent](https://arxiv.org/abs/1705.05491), *POMACS'18*              | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/geomed.py)   |
| **AutoGM**   | *Li et al.* [Byzantine-Robust Aggregation in Federated Learning Empowered Industrial IoT](https://ieeexplore.ieee.org/abstract/document/9614992), *IEEE TII'22*              | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/autogm.py)   |
| **Median**   | *Yin et al.* [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a), *ICML'18*              | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/median.py)   |
| **TrimmedMean**   | *Yin et al.* [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a), *ICML'18*              | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/trimmedmean.py)   |
| **CenteredClipping**   | *Karimireddy et al.* [Learning from History for Byzantine Robust Optimization](http://proceedings.mlr.press/v139/karimireddy21a.html), *ICML'21*              | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/centeredclipping.py)   |
| **Clustering**   | *Sattler et al.* [On the byzantine robustness of clustered federated learning](https://ieeexplore.ieee.org/abstract/document/9054676), *ICASSP'20*              | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/clustering.py)   |
| **ClippedClustering**   | *Li et al.* [An Experimental Study of Byzantine-Robust sAggregation Schemes in Federated Learning](https://www.techrxiv.org/articles/preprint/An_Experimental_Study_of_Byzantine-Robust_Aggregation_Schemes_in_Federated_Learning/19560325), *TechRxiv'22*              | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/clippedclustering.py)   |


## Customize Defense

(sec:customaggregator)=
The following example shows how to customize aggregation scheme.

```python
import ray
import torch
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet
from blades.simulator import Simulator

cifar10 = CIFAR10(num_clients=20, iid=True)  # built-in federated cifar10 dataset


class Median():
    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2
    

# configuration parameters
conf_params = {
    "dataset": cifar10,
    "aggregator": Median(),  # defense: robust aggregation
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0)
simulator = Simulator(**conf_params)
```