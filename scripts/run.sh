#!/bin/bash

# pkill -9 ray

ray start --head --port=6379

run_all_attacks() {
    for attack in "ipm" #"signflipping" "labelflipping" "alie" "noise"
    do
        for num_byzantine in 0
        do
            args="--global_round 50 --noniid --dataset cifar10 --num_gpus 4 --use-cuda --batch_size 32 --seed 0 --agg $1 --num_byzantine $num_byzantine --attack $attack"
            echo ${args}
            arg_str="\""
            for var in ${args}
                do
                    arg_str="${arg_str}, \"${var}\""ss
                done
            python cifar10.py ${args}
            # nohup python mnist.py ${args} &
        done
    done
   return 10
}

export -f run_all_attacks 


for agg in 'mean' #'trimmedmean' 'geomed' 'median' 'clippedclustering' #'mean' # 'trimmedmean' 'median' 'geomed' 'clippedclustering' 'clustering' 'centeredclipping' 'mean' 'autogm'
do
    # nohup bash -c "run_all_attacks $agg" &
    bash -c "run_all_attacks $agg"
    # run_all_attacks ${args} 
done