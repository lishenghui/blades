#!/bin/bash

# pkill -9 ray
# export CUDA_VISIBLE_DEVICES=1,2,3
ray start --head --port=6379

run_all_aggs() {
    for attack in "noise" "signflipping" "labelflipping" "alie" 
    do
        for num_byzantine in 5 #1 3 5 7 9
        do
            args="--dataset $1 --algorithm $2 --global_round $3 --local_round $4  --agg $5 --num_gpus 4 --use-cuda --batch_size 128 --seed 0 --num_byzantine $num_byzantine --attack $attack"
            echo ${args}
            python main.py ${args}
        done
    done

   return 10
}

export -f run_all_aggs 


dataset='mnist'

for agg in 'clippedclustering' 'centeredclipping' 'mean' 'trimmedmean' 'krum' 'median' 'clustering' 'geomed' 'autogm'
do 
    nohup bash -c "run_all_aggs $dataset fedavg 6000 1 $agg" &
    nohup bash -c "run_all_aggs $dataset fedavg 6000 1 $agg" &
done

