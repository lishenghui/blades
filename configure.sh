#!/usr/bin/env bash


#pushd ./data/shakespeare/ || exit
#rm -rf ./data ./meta
#./preprocess.sh -s niid --sf 0.1 -k 0 -t sample -tf 0.8 --smplseed 0

#pushd ./preprocess || exit
#python download_data.py
#popd || exit
#
#popd || exit

pushd ./data/femnist/preprocess || exit
python download_data.py
popd || exit

#pushd ./data/cifar10/preprocess || exit
#python get_cifar10.py
#popd || exit