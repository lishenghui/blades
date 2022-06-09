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

from blades.datasets import MNIST
from blades.models.mnist import DNN
from blades.simulator import Simulator

mnist = MNIST(data_root="./data", train_bs=32, num_clients=10)  # built-in federated MNIST dataset

# configuration parameters
conf_params = {
    "dataset": mnist,
    "aggregator": "trimmedmean",  # aggregation
    "num_byzantine": 3,  # number of Byzantine clients
    "attack": "alie",  # attack strategy
    "attack_param": {"num_clients": 10,  # attacker parameters
                    "num_byzantine": 3},
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0)
simulator = Simulator(**conf_params)

model = DNN()
# runtime parameters
run_params = {
    "model": model,  # global model
    "server_optimizer": 'SGD',  # ,server_opt  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 2,  # number of steps per round
    "server_lr": 1,
    "client_lr": 0.1,  # learning rate
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

install
attacks
defenses
references/api_reference
```

