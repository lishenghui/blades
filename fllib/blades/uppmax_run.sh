#!/bin/bash
#SBATCH -A snic2022-22-835
#SBATCH -t 20:00:00
#SBATCH -p core
#SBATCH -n 16
#SBATCH -M snowy
#SBATCH --gpus=2

python train.py file ./tuned_examples/fedavg_60_mnist.yaml