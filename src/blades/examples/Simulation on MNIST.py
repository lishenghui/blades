"""
Simulation on Mnist Dataset
===========================

"""

import sys, os
import ray
import json
import pandas as pd
import seaborn as sns
from blades.datasets import MNIST
from blades.models.mnist import MLP
from blades.simulator import Simulator


# In[2]:


# Get the IP address of the Spark driver node
# driver_ip = spark.sparkContext.getConf().get('spark.driver.host')

# Initialize Ray
# ray.init(ignore_reinit_error=True, address=f'{driver_ip}:9339', _redis_password='d4t4bricks')
ray.init()
# ray.init(address='auto')


# In[3]:


# %sh rm -rf /dbfs/data


# In[9]:


# mnist = MNIST(data_root="/dbfs/data", train_bs=32, num_clients=20)  # built-in federated MNIST dataset
mnist = MNIST(data_root="./data", train_bs=32, num_clients=20)  # built-in federated MNIST dataset

# configuration parameters
conf_params = {
    "dataset": mnist,
#     "aggregator": "trimmedmean",  # aggregation
    "num_byzantine": 8,  # number of Byzantine input
    "attack": "ipm",  # attack strategy
    "log_path": "dbfs/outputs",
    "attack_params": {   
                          "epsilon": 100,
                     },
    "num_actors": 1,  # number of training actors
    "seed": 1,  # reproducibility
}

run_params = {
#     "model": model,  # global model
    "server_optimizer": 'SGD',  # ,server_opt  # server optimizer
    "client_optimizer": 'SGD',  # client optimizer
    "loss": "crossentropy",  # loss function
    "global_rounds": 10,  # number of global rounds
    "local_steps": 10,  # number of steps per round
    "server_lr": 1,
    "client_lr": 0.1,  # learning rate
}

aggs = {
    'mean': {},
#     'trimmedmean': {"num_byzantine": 8},
#     'geomed': {},
#     'median': {},
#     'clippedclustering': {},
}


# In[11]:


for agg in aggs:    
    conf_params['aggregator'] = agg
    conf_params['log_path'] = f"./outputs/{agg}"
#     conf_params['log_path'] = f"dbfs/outputs/{k}"
    model = MLP()
    run_params['model'] = model
    simulator = Simulator(**conf_params)
    simulator.run(**run_params)


# In[6]:


def read_json(path):
    validation = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip().replace("'", '"')
            line = line.replace("nan", '"nan"')
            try:
                data = json.loads(line)
            except:
                print(line)
                raise
            if data['_meta']['type'] == 'test':
                validation.append(data)
    return validation

def transform(entry, agg):  
    return {
        'Round Number': entry['E'],
        'Accuracy (%)': entry['top1'],
        "Loss": entry['Loss'],
        'AGG': agg,
    }


# In[7]:


df = []
for agg in aggs:
    path = f"./outputs/{agg}/stats"
    validation_entries = read_json(path)
    df += list(map(lambda x: transform(x, agg=agg), validation_entries))
df = pd.DataFrame(df)


# In[8]:


g = sns.lineplot(
    data=df, 
    x="Round Number", y="Accuracy (%)",  
    hue="AGG",
    ci=None,
)

# g.savefig("num_byzantine.pdf", bbox_inches = "tight") 


# In[ ]:




