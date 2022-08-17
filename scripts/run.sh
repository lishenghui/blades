#!/bin/bash

# pkill -9 ray

pushd ../src || exit
  python setup.py develop
popd || exit

ray start --head --port=6379

run_all_attacks() {
    for agg in 'trimmedmean' 'median' 'clippedclustering' #'median' #'mean' 'geomed' 'autogm' 'krum' 'median' 'clustering' 'centeredclipping' 
    do
        for attack in  "noise" #"noise"  #"signflipping" "noise" "ipm" "alie" 
        do
            for num_byzantine in 5
            do
                args="--dataset $1 --algorithm $2 --num_clients 20 --global_round $3 --local_round $4 $5 --num_gpus 4 --use-cuda --batch_size 128 --seed 0 --agg $agg --num_byzantine $num_byzantine --attack $attack"
                echo ${args}
                nohup python main.py ${args} &
                # python main.py ${args}
            done
        done
    done
   return 10
}

export -f run_all_attacks 


dataset='mnist'
# nohup bash -c "run_all_attacks $dataset fedsgd 6000 1 " &
nohup bash -c "run_all_attacks $dataset fedsgd 6000 1" &
# run_all_attacks $dataset fedsgd 6000 1
# nohup bash -c "run_all_attacks $dataset fedsgd 6000 1 --noniid" &
