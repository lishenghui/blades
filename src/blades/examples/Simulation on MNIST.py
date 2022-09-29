"""
Simulation on Mnist Dataset
===========================

"""
import json

import pandas as pd
import ray
import seaborn as sns

from blades.core.simulator import Simulator
from blades.datasets import MNIST
from blades.models.mnist import MLP

# Initialize Ray
ray.init()
# ray.init(address='auto')

# mnist = MNIST(data_root="/dbfs/data", train_bs=32, num_clients=20)
# built-in federated MNIST dataset
mnist = MNIST(data_root="./data", train_bs=32, num_clients=20, seed=0)

# configuration parameters
conf_params = {
    "dataset": mnist,
    #     "aggregator": "trimmedmean",  # aggregation
    "num_byzantine": 8,  # number of Byzantine input
    "attack": "ipm",  # attack strategy
    # "log_path": "dbfs/outputs",
    "attack_kws": {
        "epsilon": 100,
    },
    "num_actors": 1,  # number of training actors
    "seed": 1,  # reproducibility
}

run_params = {
    #     "global_model": global_model,  # global global_model
    "server_optimizer": "SGD",  # ,server_opt  # server optimizer
    "client_optimizer": "SGD",  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 10,  # number of global rounds
    "local_steps": 10,  # number of steps per round
    "server_lr": 1,
    "client_lr": 0.1,  # learning rate
}

aggs = {
    "mean": {},
    "trimmedmean": {"num_byzantine": 8},
    "geomed": {},
    "median": {},
    "clippedclustering": {},
}

for agg in aggs:
    conf_params["aggregator"] = agg
    conf_params["log_path"] = f"./outputs/{agg}"
    #     conf_params['log_path'] = f"dbfs/outputs/{k}"
    model = MLP()
    run_params["global_model"] = model
    simulator = Simulator(**conf_params)
    simulator.run(**run_params)


def read_json(path):
    validation = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().replace("'", '"')
            line = line.replace("nan", '"nan"')
            try:
                data = json.loads(line)
            except IOError:
                print(line)
            if data["_meta"]["type"] == "test":
                validation.append(data)
    return validation


def transform(entry, agg):
    return {
        "Round Number": entry["Round"],
        "Accuracy (%)": entry["top1"],
        "Loss": entry["Loss"],
        "AGG": agg,
    }


df = []
for agg in aggs:
    path = f"./outputs/{agg}/stats"
    validation_entries = read_json(path)
    df += list(map(lambda x: transform(x, agg=agg), validation_entries))
df = pd.DataFrame(df)

g = sns.lineplot(
    data=df,
    x="Round Number",
    y="Accuracy (%)",
    hue="AGG",
    ci=None,
)
