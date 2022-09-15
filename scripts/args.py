import argparse
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noniid", action="store_true", default=False)
    parser.add_argument("--ipmlarge", action="store_true", default=False)
    parser.add_argument("--dp", action="store_true", default=False)
    parser.add_argument("--local_mode", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--global_round", type=int, default=400)
    parser.add_argument("--local_round", type=int, default=50)
    parser.add_argument("--serv_momentum", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--validate_interval", type=int, default=100)
    parser.add_argument('--metrics_name', help='name for metrics file;', type=str, default='none', required=False)
    parser.add_argument("--attack", type=str, default='signflipping', help="Select attack types.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset")
    parser.add_argument("--algorithm", type=str, default='fedsgd', help="Optimization algorithm, either 'fedavg' or 'fedsgd'.")
    parser.add_argument("--agg", type=str, default='clippedclustering', help="Aggregator.")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_byzantine", type=int, default=5)

    parser.add_argument("--num_actors", type=int, default=5)
    parser.add_argument("--gpu_per_actor", type=float, default=0.2)
    
    # Parameters for DP
    parser.add_argument("--privacy_delta", type=float, default=1e-6)
    parser.add_argument("--privacy_epsilon", type=float, default=1.0)
    parser.add_argument("--clip_threshold", type=float, default=0.5)
    options = parser.parse_args()
    
    if options.algorithm == "fedsgd":
        options.local_round = 1
        
    options.privacy_sensitivity = 2 * options.clip_threshold / options.batch_size
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    EXP_DIR = os.path.join(ROOT_DIR, f"outputs/{options.dataset}")
    
    options.attack_args = {
        'signflipping': {},
        'noise': {"std": 0.1},
        'labelflipping': {},
        'permutation': {},
        'attackclippedclustering': {},
        'fangattack': {},
        'ipm': {"epsilon": 100 if options.ipmlarge else 0.5},
        'alie': {"num_clients": options.num_clients, "num_byzantine": options.num_byzantine},
    }
    
    options.agg_args = {
        'trimmedmean': {"num_byzantine": options.num_byzantine},
        'median': {},
        'mean': {},
        'signguard': {},
        'geomed': {},
        'dnc': {"num_byzantine": options.num_byzantine},
        'autogm': {"lamb": 2.0},
        'clippedclustering': {"max_tau": 2.0, "signguard": True, "linkage": 'average'},
        'clustering': {},
        'centeredclipping': {},
        'krum': {"num_clients": options.num_clients, "num_byzantine": options.num_byzantine},
    }

    options.adversary_args = {
        'fangattack': {"num_byzantine": options.num_byzantine, "agg": "median"},
    }
    # options.adversary_args = {"linkage": "average"}
    # options.adversary_args = {}
    options.log_dir = (
            EXP_DIR
            + f"_{options.algorithm}"
            + f"/b{options.num_byzantine}"
            + f"_{options.attack}" + (
                "_" + "_".join([k + str(v) for k, v in options.attack_args[options.attack].items()]) if
                options.attack_args[options.attack] else "")
            + f"_{options.agg}" + (
                "_" + "_".join([k + str(v) for k, v in options.agg_args[options.agg].items()]) if options.agg_args[
                    options.agg] else "")
            + (f"_lr{options.lr}")
            + (f"_serv_momentum{options.serv_momentum}")
            + (f"_bz{options.batch_size}")
            + (f"_localround{options.local_round}")
            + ("_noniid" if options.noniid else "")
            + (f"_privacy_epsilon{options.privacy_epsilon}_clip_threshold{options.clip_threshold}" if options.dp else "")
            + f"_seed{options.seed}"
    )
    
    if not torch.cuda.is_available():
        print('Unfortunaly, we currently do not have any GPU on your machine. ')
        options.num_gpus = 0
        options.gpu_per_actor = 0
    return options


options = parse_arguments()
