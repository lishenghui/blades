---
html_meta:
  description: |
    Learn how to install the Awesome Theme for your documentation project.
---

(sec:attacks)=

# Attacks
(sec:buildinattacks)=

## Build-in Attacks

### Untargeted Attack

| Strategy          | Descriptions                                                                                                                                           | Examples                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Noise** | Put random noise to the updates. | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/noiseclient.py) |
| **Labelflipping** | *Fang et al.* [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](https://www.usenix.org/conference/usenixsecurity20/presentation/fang), *USENIX Security' 20* | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/labelflippingclient.py) |
| **Signflipping** | *Li et al.* [RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets](https://ojs.aaai.org/index.php/AAAI/article/view/3968), *AAAI' 19* | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/signflippingclient.py) |
| **ALIE** | *Baruch et al.* [A little is enough: Circumventing defenses for distributed learning](https://proceedings.neurips.cc/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html), *NeurIPS' 19* | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/alieclient.py) |
| **IPM** | *Xie et al.* [Fall of empires: Breaking byzantine- tolerant sgd by inner product manipulation](https://arxiv.org/abs/1903.03936), *UAI' 20* | [[**Example**]](https://github.com/bladesteam/blades/blob/master/src/blades/attackers/ipmclient.py) |


## Customize Attacks
(sec:customattacks)=
The following example shows how to customize attack strategy.

```python
import ray
from blades.client import ByzantineClient
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet
from blades.simulator import Simulator


cifar10 = CIFAR10(num_clients=20, iid=True)  # built-in federated cifar10 dataset


class MaliciousClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super(ByzantineClient).__init__(*args, **kwargs)
    
    def omniscient_callback(self, simulator):
        updates = []
        for w in simulator._clients:
            is_byzantine = w.get_is_byzantine()
            if not is_byzantine:
                updates.append(w.get_update())
        self._save_update(-100 * (sum(updates)) / len(updates))


# configuration parameters
conf_params = {
    "dataset": cifar10,
    "aggregator": "mean",  # defense: robust aggregation
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0)
simulator = Simulator(**conf_params)

attackers = [MaliciousClient() for _ in range(5)]
simulator.register_attackers(attackers)

# runtime parameters
run_params = {
    "model": CCTNet(),  # global model
    "server_optimizer": 'SGD',  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 2,  # number of steps per round
    "lr": 0.1,  # learning rate
}
simulator.run(**run_params)

```