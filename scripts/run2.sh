#!/bin/bash

# pkill -9 ray

ray start --head --port=6379

run_all_attacks1() {
    for agg in 'krum' 'median' 'clippedclustering' 'trimmedmean' 'clustering'
    do
        for attack in "ipm" #"noise" "signflipping" "labelflipping" "alie" 
        do
            for num_byzantine in 0
            do
                args="--dataset $1 --algorithm $2 --ipmlarge --global_round $3 --local_round $4 $5 --num_gpus 4 --use-cuda --batch_size 32 --seed 0 --agg $agg --num_byzantine $num_byzantine --attack $attack"
                echo ${args}
                python main.py ${args}
            done
        done
    done
   return 10
}

export -f run_all_attacks1


dataset='mnist'
nohup bash -c "run_all_attacks1 $dataset fedavg 600 50 " &
nohup bash -c "run_all_attacks1 $dataset fedavg 600 50 --noniid " &
# sleep 5
nohup bash -c "run_all_attacks1 $dataset fedsgd 6000 1 " &
nohup bash -c "run_all_attacks1 $dataset fedsgd 6000 1 --noniid" &

run_all_attacks2() {
    for agg in 'centeredclipping' 'mean' 'geomed' 'autogm'
    do
        for attack in "ipm" #"noise" "signflipping" "labelflipping" "alie" 
        do
            for num_byzantine in 0
            do
                args="--dataset $1 --algorithm $2 --ipmlarge --global_round $3 --local_round $4 $5 --num_gpus 4 --use-cuda --batch_size 32 --seed 0 --agg $agg --num_byzantine $num_byzantine --attack $attack"
                echo ${args}
                python main.py ${args}
            done
        done
    done
   return 10
}

export -f run_all_attacks2


dataset='mnist'
nohup bash -c "run_all_attacks2 $dataset fedavg 600 50 " &
nohup bash -c "run_all_attacks2 $dataset fedavg 600 50 --noniid " &
# sleep 5
nohup bash -c "run_all_attacks2 $dataset fedsgd 6000 1 " &
nohup bash -c "run_all_attacks2 $dataset fedsgd 6000 1 --noniid" &
