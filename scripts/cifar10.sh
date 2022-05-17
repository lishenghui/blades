#!/bin/bash


cuda=2
for seed in 0
do
    for num_byzantine in 5
    do
        for attack in "IPM" #"ALIE" "IPM" "LF" "IPM_large" "IPM"  # "ALIE" "IPM" 
        do  
            for agg in "clustering" #"tm" "krum" "cm" "cp" "rfa" "autogm" "clippedclustering" 
            do
                export CUDA_VISIBLE_DEVICES=$(((cuda)))
                cuda=$(((cuda + 1) % 4))
                args="--round 2 --use-cuda --batch_size 32 --seed $seed --agg ${agg} --momentum 0.0 --num_byzantine ${num_byzantine} --attack $attack"
                # args="--round 50 --use-cuda --batch_size 32 --seed $seed --agg ${agg} --momentum 0 --num_byzantine ${num_byzantine} --attack $attack --fedavg"
                echo ${args}
                arg_str="\""
                for var in ${args}
                    do
                        arg_str="${arg_str}, \"${var}\""ss
                    done
                # echo ${arg_str}
                # echo ${cuda}
                # python main.py ${args}
                nohup python cifar10-all.py ${args} &
            done
        done
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    unset pids
done
