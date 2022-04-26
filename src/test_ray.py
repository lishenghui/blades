# -*- coding: utf-8 -*-
import time
import ray
ray.init(address="147.8.182.160:6379")

def  f1():
    time.sleep(1)

@ray.remote
def f2():
    time.sleep(1)

#以下需要十秒。
time1=time.time()
[ f1() for _ in range(50)]
print(time.time()-time1)

#以下需要一秒(假设系统至少有10个CPU)。
time2=time.time()
ray.get([ f2.remote() for _ in range(50)])
print(time.time()-time2)
