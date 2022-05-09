# -*- coding: utf-8 -*-
import time

import ray

if not ray.is_initialized():
    ray.init(include_dashboard=True, num_gpus=0)
    # ray.init(include_dashboard=False, num_gpus=args.num_gpus)


def f1():
    time.sleep(1)


@ray.remote
def f2():
    time.sleep(1)


# 以下需要十秒。
time1 = time.time()
[f1() for _ in range(4)]
print(time.time() - time1)

# 以下需要一秒(假设系统至少有10个CPU)。
time2 = time.time()
ray.get([f2.remote() for _ in range(4)])
print(time.time() - time2)
