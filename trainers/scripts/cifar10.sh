#!/bin/bash


cuda=0
for seed in 0
do
    for num_byzantine in 0
    do
        for attack in "ALIE" # "IPM" "LF" "IPM_large"  "ALIE"  "IPM" "ALIE" # "ALIE" "IPM" 
        do  
            for agg in "clippedclustering" #"rfa" "tm" #"clustering" #"clippedclustering" #"krum" "avg" "autogm"  "cp" "tm" "cm" 
            do
                export CUDA_VISIBLE_DEVICES=$(((cuda)))
                cuda=$(((cuda + 1) % 4))
                args="--round 50 --use-cuda --batch_size 32 --seed $seed --agg ${agg} --momentum 0 --num_byzantine ${num_byzantine} --attack $attack --fedavg"
                echo ${args}
                arg_str="\""
                for var in ${args}
                    do
                        arg_str="${arg_str}, \"${var}\""
                    done
                # echo ${arg_str}
                # echo ${cuda}
                python cifar10-all.py ${args}
                # nohup python cifar10-all.py ${args} &
            done
        done
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
done
