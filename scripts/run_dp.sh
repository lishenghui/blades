#!/bin/bash

# pkill -9 ray
# export CUDA_VISIBLE_DEVICES=1,2,3
ray start --head --port=6379

pushd ../src || exit
  python setup.py develop
popd || exit

for privacy_epsilon in 100.0 #5.0 10.0 20
  do
    args="--privacy_epsilon $privacy_epsilon --dp --num_clients 50 --global_round 6000 --num_byzantine 5 --dataset cifar10 --agg median --serv_momentum 0.9 --batch_size 128 --attack labelflipping"
    python main.py ${args}
    # nohup python dp_cpu.py ${args} &
    # python dp_cpu.py ${args}
  done