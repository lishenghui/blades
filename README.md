# Byzantine-Robust AFederated Learning

This repository contains the code and experiments for the paper:

[An Experimental Study of Byzantine-Robust Aggregation Schemes in Federated Learning](https://www.techrxiv.org/articles/preprint/An_Experimental_Study_of_Byzantine-Robust_Aggregation_Schemes_in_Federated_Learning/19560325)

## Datasets

1. Bosch
  * **Overview:** Image Dataset. See [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  * **Details:** 2 different classes.
  * **Task:** Binary Classification


2. FEMNIST

  * **Overview:** Image Dataset
  * **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
  * **Task:** Image Classification


## Notes

- Install the libraries listed in ```requirements.txt```
    - I.e. with pip: run ```pip3 install -r requirements.txt```
    

## Ref

### LEAF benchmark
* **Homepage:** [leaf.cmu.edu](https://leaf.cmu.edu)
* **Paper:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)
