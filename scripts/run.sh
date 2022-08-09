#!/bin/bash

# pkill -9 ray

ray start --head --port=6379

run_all_attacks() {
    for agg in 'mean' 'trimmedmean' 'geomed' 'median' 'clippedclustering' 'clustering' 'centeredclipping' 'autogm'
    do
        for attack in "ipm" "signflipping" "labelflipping" "alie" "noise"
        do
            for num_byzantine in 5
            do
                args="--dataset $1 --algorithm $2 --global_round $3 --local_round $4 $5 --num_gpus 4 --use-cuda --batch_size 32 --seed 0 --agg $agg --num_byzantine $num_byzantine --attack $attack"
                echo ${args}
                python main.py ${args}
            done
        done
    done
   return 10
}

export -f run_all_attacks 


nohup bash -c "run_all_attacks mnist fedavg 600 50 " &
nohup bash -c "run_all_attacks mnist fedavg 600 50 --noniid " &

sleep 5
nohup bash -c "run_all_attacks mnist fedsgd 6000 1 " &
nohup bash -c "run_all_attacks mnist fedsgd 6000 1 --noniid" &
