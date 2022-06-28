"""
A mini example
===========================

"""

import ray
from blades.datasets import MNIST
from blades.models.mnist import MLP
from blades.simulator import Simulator



# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"


mnist = MNIST(data_root="./data", train_bs=32, num_clients=10)  # built-in federated MNIST dataset

# configuration parameters
conf_params = {
    "dataset": mnist,
    "aggregator": "mean",  # aggregation
    "num_byzantine": 0,  # number of Byzantine input
    "attack": "alie",  # attack strategy
    "attack_params": {"num_clients": 1,  # attacker parameters
                     "num_byzantine": 0},
    "num_actors": 10,  # number of training actors
    "use_cuda": False,
    "gpu_per_actor": 0.,
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0, local_mode=False)
simulator = Simulator(**conf_params)

model = MLP()
# runtime parameters
run_params = {
    "model": model,  # global model
    "server_optimizer": 'SGD',  # ,server_opt  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 50,  # number of steps per round
    "server_lr": 1.0,
    "client_lr": 0.1,  # learning rate
}
simulator.run(**run_params)
