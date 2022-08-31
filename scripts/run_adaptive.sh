#!/bin/bash

# pkill -9 ray
# export CUDA_VISIBLE_DEVICES=1,2,3
ray start --head --port=6379

pushd ../src || exit
  python setup.py develop
popd || exit

run_all_aggs() {
    for attack in "attackclippedclustering"
    do  
        serv_momentum=0.9
        for num_byzantine in 7
        do
            args="--dataset $1 --algorithm $2 --global_round $3 --local_round $4  --agg $5 --num_gpus 4 --num_byzantine $num_byzantine --use-cuda --batch_size $6 $7 --seed 0 --serv_momentum $serv_momentum --attack $attack"
            echo ${args}
            python main.py ${args}
        done
    done
   return 10
}

export -f run_all_aggs 


dataset='cifar10'

for agg in 'clippedclustering' #'median' 'trimmedmean' 'centeredclipping' 'mean' 'krum' 'clustering' 'geomed' 'autogm'
do 
    # nohup bash -c "run_all_aggs $dataset fedsgd 6000 1 $agg 64 --noniid" &
    # nohup bash -c "run_all_aggs $dataset fedavg 600 50 $agg 64 --noniid" &
    nohup bash -c "run_all_aggs $dataset fedavg 600 50 $agg 64" &
done

dataset='mnist'

for agg in 'clippedclustering' #'median' 'trimmedmean' 'centeredclipping' 'mean' 'krum' 'clustering' 'geomed' 'autogm'
do 
    nohup bash -c "run_all_aggs $dataset fedsgd 6000 1 $agg 128" &
    # nohup bash -c "run_all_aggs $dataset fedsgd 6000 1 $agg 128 --noniid" &
    # nohup bash -c "run_all_aggs $dataset fedavg 600 50 $agg 128 --noniid" &
done
