import ray

from blades.datasets import MNIST
from blades.models.mnist import DNN
from blades.simulator import Simulator

mnist = MNIST(data_root="./data", train_bs=32, num_clients=10)  # built-in federated MNIST dataset

# configuration parameters
conf_params = {
    "dataset": mnist,
    "aggregator": "mean",  # aggregation
    "num_byzantine": 0,  # number of Byzantine input
    "attack": "alie",  # attack strategy
    "attack_param": {"num_clients": 1,  # attacker parameters
                     "num_byzantine": 0},
    "num_actors": 4,  # number of training actors
    "seed": 1,  # reproducibility
}

ray.init(num_gpus=0, local_mode=True)
simulator = Simulator(**conf_params)

model = DNN()
# runtime parameters
run_params = {
    "model": model,  # global model
    "server_optimizer": 'SGD',  # ,server_opt  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 400,  # number of global rounds
    "local_steps": 1,  # number of steps per round
    "server_lr": 0.1,
    "client_lr": 1.0,  # learning rate
}
simulator.run(**run_params)
