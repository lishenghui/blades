"""
Customization of attack strategy
=================================

To customize attack strategies, you only need to subclass ``ByzantineClient``
and override its methods. At present, there are three methods for the
customization of attack strategies, i.e.,

- ``train_global_model``:
    You can customize the local training process and do whatever you want.
    For example, flipping the
    sign of gradients at each step.
- ``on_train_batch_begin``:
    This method is called right before each batch, making it possible to modify
    the batch data for updating.
- ``omniscient_callback``:
 This method is called after local optimization. By overriding it,
 the attacker can have full knowledge of the whole system (e.g., updates from
  all input), so that it can adjust the global_model update accordingly. This method
   is especially useful for adaptive attacks.
"""

import ray
import torch

from blades.clients.client import ByzantineClient
from blades.core.simulator import Simulator
from blades.datasets import MNIST
from blades.models.mnist import MLP

# built-in federated MNIST dataset
mnist = MNIST(data_root="./data", train_bs=32, num_clients=10)


# Subclass the ``ByzantineClient``
class MaliciousClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 10

    # Attack by flipping the sign of gradient, which is equivalent
    # to stochastic gradient ascent.
    def train_global_model(self, data_batches, opt):
        for data, target in data_batches:
            data, target = data.to(self.device), target.to(self.device)
            data, target = self.on_train_batch_begin(data=data, target=target)
            opt.zero_grad()

            output = self.global_model(data)
            loss = torch.clamp(self.loss_func(output, target), 0, 1e5)
            loss.backward()
            for name, p in self.global_model.named_parameters():
                p.grad.data = -p.grad.data
            opt.step()

    # Attack by flipping the labels of training samples.
    def on_train_batch_begin(self, data, target, logs=None):
        return data, self.num_classes - 1 - target

    # Access the updates from all honest clients and design
    # malicious updates accordingly.
    def omniscient_callback(self, simulator):
        updates = []
        for w in simulator.get_clients():
            if not w.is_byzantine():
                updates.append(w.get_update())
        self.save_update(-100 * (sum(updates)) / len(updates))


# configuration parameters
conf_params = {
    "dataset": mnist,
    "aggregator": "clippedclustering",  # defense: robust aggregation
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0, local_mode=True)
simulator = Simulator(**conf_params)

# %%
# Register attacks in the simulator.

attackers = [MaliciousClient() for _ in range(5)]
# By default, the first five clients will be replaced.
simulator.register_attackers(attackers)

# %%
# Configure run time parameters and run the experiment.

run_params = {
    "global_model": MLP(),  # global global_model
    "server_optimizer": "SGD",  # server optimizer
    "client_optimizer": "SGD",  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 50,  # number of steps per round
    "client_lr": 0.1,  # learning rate
}
simulator.run(**run_params)
