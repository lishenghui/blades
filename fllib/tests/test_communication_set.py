import time

import torch

from fllib.communication.communicator_set import CommunicationSet

tensor = torch.ones(464154).to("cuda:0")

com = CommunicationSet(num_workers=4, num_gpus_per_worker=0.95)
t_s = time.time()
com.broadcast(tensor)
t_1 = time.time()
print("time: ", t_1 - t_s)

t_s = time.time()
com.broadcast(tensor)
t_1 = time.time()
print("time" * 10, t_1 - t_s)

t_s = time.time()
com.broadcast(tensor)
t_1 = time.time()
print("time" * 10, t_1 - t_s)

t_s = time.time()
com.broadcast(tensor)
t_1 = time.time()
print("time" * 10, t_1 - t_s)

t_s = time.time()
com.broadcast(tensor * 5)
t_1 = time.time()
print("time" * 10, t_1 - t_s)
