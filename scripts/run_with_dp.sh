#!/bin/bash

# pkill -9 ray
# export CUDA_VISIBLE_DEVICES=1,2,3
ray start --head --port=6379

pushd ../src || exit
  python setup.py develop
popd || exit


run_one_agg() {
for privacy_epsilon in 1.0 #5.0 10.0
  do
    for num_byzantine in 5 #10
    do
      args="
            --agg $1 \
            --dp_privacy_epsilon $privacy_epsilon \
            --dp \
            --num_clients 50 \
            --global_round 6000 \
            --num_byzantine $num_byzantine \
            --dataset cifar10 \
            --serv_momentum 0.9 \
            --batch_size 64 \
            --attack alie \
            "
      python main.py ${args}
    done
  done

  for privacy_epsilon in 100.0
  do
    for num_byzantine in 0 #5 10
    do
      args="
            --agg $1 \
            --dp_privacy_epsilon $privacy_epsilon \
            --num_clients 50 \
            --global_round 6000 \
            --num_byzantine $num_byzantine \
            --dataset cifar10 \
            --serv_momentum 0.9 \
            --batch_size 64 \
            --attack alie \
            "
      python main.py ${args}
    done
  done
}

export -f run_one_agg
 

cuda=0
for agg in 'clippedclustering' 'multikrum' 'dnc' 'clustering'
do 
    export CUDA_VISIBLE_DEVICES=$(((cuda + 1)))
    cuda=$(((cuda + 1) % 3))
    nohup bash -c "run_one_agg $agg" &
done