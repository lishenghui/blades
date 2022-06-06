---
html_meta:
  description: Create functional and beautiful websites for your documentation with Sphinx and the Awesome Sphinx Theme.
  keywords: Documentation, Sphinx, Python
---

<!-- vale Google.Headings = NO -->

# Welcome to Blades documentation

<!-- vale Google.Headings = YES -->

```{rst-class} lead
A simulator for Byzantine-robust federated Learning with Attacks and Defenses Experimental Simulation
```

---

## Get started

1. Install the simulator:

   ```terminal
   pip install blades
   ```

   ```{seealso}
   {ref}`sec:install`
   ```

2. Test ``Blades`` with the following example:

```python
import ray

from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet
from blades.simulator import Simulator

cifar10 = CIFAR10(num_clients=20, iid=True)  # built-in federated cifar10 dataset

# configuration parameters
conf_params = {
    "dataset": cifar10,
    "aggregator": "mean",  # defense: robust aggregation
    "num_byzantine": 5,  # number of byzantine clients
    "attack": "alie",  # attack strategy
    "attack_para": {"n": 20,  # attacker parameters
                    "m": 5},
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0)
simulator = Simulator(**conf_params)

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

```{seealso}
{ref}`sec:attacks`
```

In the _attacks_ section, you can learn more about the 
{ref}`Build-in Attacks <sec:buildinattacks>` and
{ref}`Customize Attacks <sec:customattacks>`.


## Give feedback

Is something broken or missing?
Create a [GitHub issue](https://github.com/bladesteam/blades/issues/new).

<!-- vale Google.Headings = NO -->
<!-- vale 18F.Headings = NO -->

```{toctree}
---
hidden: true
caption: Documentation
glob: true
---

attacks
defenses
references/api_reference
```

