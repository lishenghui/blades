#!/bin/bash

# pkill -9 ray

# ray start --head --port=6379

for seed in 0
do
    for num_byzantine in 8
    do
        for attack in "ipm" #"signflipping" "labelflipping" "alie" "noise"
        do  
            for agg in 'trimmedmean' 'geomed' 'median' #'clippedclustering' #'mean' # 'trimmedmean' 'median' 'geomed' 'clippedclustering' 'clustering' 'centeredclipping' 'mean' 'autogm'
            do
                args="--global_round 600 --dataset mnist --num_gpus 3 --use-cuda --batch_size 32 --seed $seed --agg ${agg} --num_byzantine ${num_byzantine} --attack $attack"
                echo ${args}
                arg_str="\""
                for var in ${args}
                    do
                        arg_str="${arg_str}, \"${var}\""ss
                    done
                # python mnist.py ${args}
                nohup python mnist.py ${args} &
            done
        done
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
done
