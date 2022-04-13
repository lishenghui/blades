#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash mnist-dissensus-attack.sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 


cuda=0
function grid_search_3 {
    for seed in 0
    do
        for num_byzantine in 8
        do
            for attack in "IPM_large" #"BF" "Noise" "IPM" "LF" "IPM_large" "IPM"
            do  
                for agg in "clustering" #"autogm" "rfa" "cp" "tm" "cm" "clustering"  "krum" "avg" "clippedclustering" #
                do
                    
                    export CUDA_VISIBLE_DEVICES=$(((cuda + 1)))
                    cuda=$(((cuda + 1) % 3))
                    args="--round 150 --use-cuda --batch_size 512--seed $seed --agg ${agg} --momentum 0 --num_byzantine ${num_byzantine} --attack $attack"
                    echo ${args}
                    arg_str="\""
                    for var in ${args}
                        do
                            arg_str="${arg_str}, \"${var}\""
                        done
                    echo ${arg_str}
                    echo ${cuda}
                    # python cifar10-all.py ${args}
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
}


grid_search_3
            


