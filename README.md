# ⚔🛡 **Blades**: A simulator for Byzantine-robust federated Learning with Attacks and Defenses Experimental Simulation

<!-- <p align="center">
  <img width = "450" height = "150" src="https://github.com/
" alt="banner"/>
  <br/>
</p> -->

<p align=center>
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/badge/Python->=3.9-3776AB?logo=python" alt="Python">
  </a>    
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.8-FF6F00?logo=pytorch" alt="pytorch">
  </a>   
  <!-- <a href="https://pypi.org/project/graphwar/">
    <img src="https://badge.fury.io/py/graphwar.png" alt="pypi">
  </a>        -->
  <a href="https://github.com/EdisonLeeeee/GraphWar/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EdisonLeeeee/GraphWar" alt="license">
    <img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/>    
  </a>       
</p>
                                                                   

> Know thy self, know thy enemy. A thousand battles, a thousand victories.
> 
> 「知己知彼，百战百胜」 ——《孙子兵法•谋攻篇》


NOTE: Blade is still in the early stages and the API are subject to change.
If you are interested in this project, don't hesitate to contact me or make a PR directly.

# 🚀 Installation

Please make sure you have installed [PyTorch](https://pytorch.org) and [Ray](https://docs.ray.io/en/latest/).


```bash
# Coming soon
pip install blades
```

<!-- or

```bash
# Recommended
git clone https://github.com/EdisonLeeeee/GraphWar.git && cd GraphWar
pip install -e . --verbose
``` -->

<!-- where `-e` means "editable" mode so you don't have to reinstall every time you make changes. -->

# ⚡ Get Started


## How fast can we train and evaluate your own GNN?
Take `IPM Attack` and `Krum Aggregation` as an example:
```python
import ray
from blades.simulator import Simulator
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet

cifar10 = CIFAR10(num_clients=20, iid=True) # built-in federated cifar10 dataset

# configuration parameters
conf_params = {
    "dataset": cifar10,
    "aggregator": "Krum",   # defense: robust aggregation
    "num_byzantine": 5,     # number of byzantine clients
    "attack": "alie",       # attack strategy
    "attack_para":{"n": 20, # attacker parameters
                   "m": 5},
    "num_actors": 4,        # number of training actors
    "seed": 1,              # reproducibility
}

ray.init(num_gpus=0)
simulator = Simulator(**conf_params)

# runtime parameters
run_params = {
    "model": CCTNet(),         # global model
    "server_optimizer": 'SGD', # server optimizer
    "client_optimizer": 'SGD', # client optimizer
    "loss": "crossentropy",    # loss function
    "global_rounds": 400,      # number of global rounds
    "local_steps": 2,         # number of steps per round
    "lr": 0.1,                 # learning rate
}
simulator.run(**run_params)
```


# 👀 Implementations

In detail, the following methods are currently implemented:

## ⚔ Attack

#### Untargeted Attack

| Methods          | Descriptions                                                                                                                                           | Examples                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **NoiseAttack** | Put random noise to the updates. | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/noiseclient.py) |
| **LabelflippingAttack** | *Fang et al.* [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://www.usenix.org/conference/usenixsecurity20/presentation/fang), *USENIX Security' 20* | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/labelflippingclient.py) |
| **SignflippingAttack** | *Li et al.* [RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets](https://ojs.aaai.org/index.php/AAAI/article/view/3968), *AAAI' 19* | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/signflippingclient.py) |
| **ALIEAttack** | *Baruch et al.* [A little is enough: Circumventing defenses for distributed learning](https://proceedings.neurips.cc/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html), *NeurIPS' 19* | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/alieclient.py) |
| **IPMAttack** | *Xie et al.* [Fall of empires: Breaking byzantine- tolerant sgd by inner product manipulation](https://arxiv.org/abs/1903.03936), *UAI' 20* | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/ipmclient.py) |






## 🛡 Defense

### Robust Aggregation

| Methods   | Descriptions                                                                                                                               | Examples                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **Krum**   | *Blanchard et al.* [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html), *NIPS'17*              | [[**Example**]](https://github.com/EdisonLeeeee/GraphWar/blob/master/examples/models/gcn.py)   |
| **GeoMed**   | *Blanchard et al.* [Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent](https://arxiv.org/abs/1705.05491), *NIPS'17*              | [[**Example**]](https://github.com/EdisonLeeeee/GraphWar/blob/master/examples/models/gcn.py)   |

<!-- 
# ❓ Known Issues
+ Despite our best efforts, we still had difficulty reproducing the results of [GNNGUARD](https://arxiv.org/abs/2006.08149) in the paper. If you find any problems, please don't hesitate to contact me.
+ Untargeted attacks are suffering from performance degradation, as also in DeepRobust, when a validation set is used during training with model picking. Such phenomenon has also been revealed in [Black-box Gradient Attack on Graph Neural Networks: Deeper Insights in Graph-based Attack and Defense](https://arxiv.org/abs/2104.15061). -->
