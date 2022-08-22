#!/bin/bash

# pkill -9 ray
# export CUDA_VISIBLE_DEVICES=1,2,3
ray start --head --port=6379

pushd ../src || exit
  python setup.py develop
popd || exit

run_all_aggs() {
    serv_momentum=0.9
    for attack in "noise"
    do
        for batch_size in 2500 512 128 
        do
            args="--dataset $1 --algorithm $2 --global_round $3 --local_round $4  --agg $5 $6 --num_gpus 4 --num_byzantine 5 --use-cuda --batch_size $batch_size --seed 0 --serv_momentum $serv_momentum --attack $attack"
            echo ${args}
            python main.py ${args}
        done
    done
   return 10
}

export -f run_all_aggs 


dataset='cifar10'

for agg in 'clippedclustering' 'median' 'trimmedmean' 'centeredclipping' 'clustering' 'geomed' 'autogm' 'mean' 'krum'
do 
    nohup bash -c "run_all_aggs $dataset fedsgd 6000 1 $agg" &
    # nohup bash -c "run_all_aggs $dataset fedavg 600 50 $agg  --noniid" &
done

