"""
A mini example
===========================

"""

import ray

from blades.core.simulator import Simulator
from blades.datasets import MNIST
from blades.models.mnist import MLP

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

# built-in federated MNIST dataset
mnist = MNIST(data_root="./data", train_bs=32, num_clients=10)

# configuration parameters
conf_params = {
    "dataset": mnist,
    "aggregator": "mean",  # aggregation
    "num_byzantine": 4,  # number of Byzantine input
    "attack": "alie",  # attack strategy
    "attack_kws": {"num_clients": 10, "num_byzantine": 4},  # attacker parameters
    "num_actors": 4,  # number of training actors
    # "num_actors": 10,  # number of training actors
    "use_cuda": False,
    "gpu_per_actor": 0.0,
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0, local_mode=False)
simulator = Simulator(**conf_params)

model = MLP()
# runtime parameters
run_params = {
    "global_model": model,  # global global_model
    "server_optimizer": "SGD",  # ,server_opt  # server optimizer
    "client_optimizer": "SGD",  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 100,  # number of global rounds
    "local_steps": 50,  # number of steps per round
    "server_lr": 1.0,
    "client_lr": 0.1,  # learning rate
}
simulator.run(**run_params)
