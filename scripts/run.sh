#!/bin/bash

# pkill -9 ray

ray start --head --port=6379

run_all_attacks() {
    for agg in 'clippedclustering' #'mean' 'geomed' 'autogm' 'krum' 'median' 'clustering' 'centeredclipping' 
    do
        for attack in  "noise"  #"signflipping" "noise" "ipm" "alie" 
        do
            for num_byzantine in 0
            do
                args="--dataset $1 --algorithm $2 --global_round $3 --local_round $4 $5 --num_gpus 4 --use-cuda --batch_size 32 --seed 0 --agg $agg --num_byzantine $num_byzantine --attack $attack"
                echo ${args}
                nohup python main.py ${args} &
                # python main.py ${args}
            done
        done
    done
   return 10
}

export -f run_all_attacks 


dataset='cifar10'
# nohup bash -c "run_all_attacks $dataset fedsgd 6000 1 " &
nohup bash -c "run_all_attacks $dataset fedsgd 6000 1" &
# run_all_attacks $dataset fedsgd 6000 1
# nohup bash -c "run_all_attacks $dataset fedsgd 6000 1 --noniid" &
