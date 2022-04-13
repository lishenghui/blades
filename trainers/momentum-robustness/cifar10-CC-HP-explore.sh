#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash mnist-dissensus-attack.sh


function grid_search {
    for seed in {0..2}
    do
        for attack in "LF" "BF"
        do
            for tau in 1e-1 1e1 1e3
            do
                for inner in 1 3 5
                do
                    python cifar10-CC-HP-explore.py --use-cuda --seed $seed --inner-iterations $inner --tau $tau --attack $attack &
                    pids[$!]=$!
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

PS3='Please enter your choice: '
options=("debug" "run" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "debug")
            python cifar10-CC-HP-explore-BF-debug.py --use-cuda --seed 0 --inner-iterations 1 --tau 100 --attack "IPM" --debug
            ;;

        "run")
            grid_search
            ;;

        "Quit")
            break
            ;;

        *) echo "invalid option $REPLY";;
    esac
done


